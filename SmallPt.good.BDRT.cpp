// SmallPt.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "base.h"
#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2009
#include <stdlib.h> // Make : g++ -O3 -fopenmp explicit.cpp -o explicit
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include"lodepng.h"

int global_en_bdrt = 0;

struct Vec {        // Usage: time ./explicit 16 && xv image.ppm
	double x, y, z;                  // position, also color (r,g,b)
	Vec(double x_ = 0, double y_ = 0, double z_ = 0) { x = x_; y = y_; z = z_; }
	Vec operator+(const Vec& b) const { return Vec(x + b.x, y + b.y, z + b.z); }
	Vec operator-(const Vec& b) const { return Vec(x - b.x, y - b.y, z - b.z); }
	Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }
	Vec mult(const Vec& b) const { return Vec(x * b.x, y * b.y, z * b.z); }
	Vec& norm() { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }
	double dot(const Vec& b) const { return x * b.x + y * b.y + z * b.z; } // cross:
	Vec operator%(Vec& b) { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
};
//https://blog.csdn.net/aa20274270/article/details/52709444
void hit_uv_sphere(Vec hit_point, Vec pos, double r, double &u, double &v)
{
	Vec loc_hit = (hit_point - pos) * (1 / r);
	double theta = acos(loc_hit.y);
	double phi = atan2(loc_hit.x, loc_hit.z);
	if (phi < 0.0)
	{
		phi += M_PI * 2;
	}
	u = phi * (1 / (M_PI * 2));
	v = 1 - theta * M_1_PI;
}
class mytexture {
public:
	// Load file and decode image.
	std::vector<unsigned char> image;
	unsigned width, height;
	int valid;
	mytexture() { valid = 0; }
	int chn;
	void load_image(const char* path)
	{
		std::string path2 = path;
		unsigned error = lodepng::decode(image, width, height, path2);
		valid = 1;
		chn = 4;
	}
	Vec sample(double u, double v, int base = 4)
	{
		if (!valid) return Vec(0);
		int wx = (double)width * u;
		int vx = (double)height * v;
		unsigned char r = image[(wx + vx * width) * chn];
		unsigned char g = image[(wx + vx * width) * chn + 1];
		unsigned char b = image[(wx + vx * width) * chn + 2];
		Vec new_col((float)r / 255.0 * base, (float)g / 255.0 * base, (float)b / 255.0 * base);
		return new_col;
	}
};
//
struct Ray { Vec o, d; Ray(Vec o_, Vec d_) : o(o_), d(d_) {} };
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()
struct Sphere {
	double rad;       // radius
	Vec p, e, c;      // position, emission, color
	Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
	int gloss;
	mytexture* tex;
	Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_, int gloss_in, mytexture*ttex = 0) :
		rad(rad_), p(p_), e(e_), c(c_), refl(refl_), gloss(gloss_in), tex(ttex) {}
	double intersect(const Ray& r) const { // returns distance, 0 if nohit
		Vec op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		double t, eps = 1e-4, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
		if (det < 0) return 0; else det = sqrt(det);
		return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
	}
};

mytexture light_tex;
mytexture earth_tex;

Vec get_obj_e(const Sphere& obj, Vec hit_p)
{
	if (obj.e.x > 0 && obj.tex)
	{
		double u, v;
		hit_uv_sphere(hit_p, obj.p, obj.rad, u, v);
		return obj.tex->sample(u, v, obj.e.x);
	}
	else
	{
		return obj.e;
	}
}
Vec get_obj_c(const Sphere& obj, Vec hit_p)
{
	if (obj.tex && obj.e.x == 0)
	{
		double u, v;
		hit_uv_sphere(hit_p, obj.p, obj.rad, u, v);
		return obj.tex->sample(u, v, 1.0);
	}
	else
	{
		return obj.c;
	}
}
#if 1
Sphere spheres[] = {//Scene: radius, position, emission, color, material
	Sphere(1e5, Vec(1e5 + 1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF,0),//Left
	Sphere(1e5, Vec(-1e5 + 99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF,0),//Rght
	Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.75,.75),DIFF,0),//Back
	Sphere(1e5, Vec(50,40.8,-1e5 + 170), Vec(),Vec(),           DIFF,0),//Frnt
	Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF,0),//Botm
	Sphere(1e5, Vec(50,-1e5 + 81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF,0),//Top
	//Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(1,1,1) * .999, SPEC,36),//Mirr - SPEC roughtness 0-100
	Sphere(16.5,Vec(27,40,20),       Vec(),Vec(1,1,1) * .999, REFR,0),
	Sphere(16.5,Vec(50,20,40),       Vec(),Vec(1,1,1) * .999, REFR,0),
	Sphere(16.5,Vec(18,16.5,47),       Vec(),Vec(1,1,1) * .999, REFR,36),//Mirr - SPEC roughtness 0-100
	Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,1,1) * .999, REFR,0),//Glas
	//Sphere(1.5, Vec(50,81.6 - 16.5,81.6),Vec(4,4,4) * 100,  Vec(), DIFF,0,0),//&light_tex),//Lite																																			
	Sphere(0.5, Vec(27,16.5,20),Vec(4,4,4) * 600,  Vec(), DIFF,0,0),//Lite
};
#else
Sphere spheres[] = {//Scene: radius, position, emission, color, material
	Sphere(1e5, Vec(1e5 + 1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF,0),//Left
	Sphere(1e5, Vec(-1e5 + 99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF,0),//Rght
	Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.75,.75),DIFF,0),//Back
	Sphere(1e5, Vec(50,40.8,-1e5 + 170), Vec(),Vec(),           DIFF,0),//Frnt
	Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF,0),//Botm
	Sphere(1e5, Vec(50,-1e5 + 81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF,0),//Top
	//Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(1,1,1) * .999, SPEC,36),//Mirr - SPEC roughtness 0-100
	Sphere(16.5,Vec(27,40,20),       Vec(),Vec(1,1,1) * .999, SPEC,0),
	Sphere(16.5,Vec(50,20,40),       Vec(),Vec(1,1,1) * .999, SPEC,0),
	Sphere(16.5,Vec(18,16.5,47),       Vec(),Vec(1,1,1) * .999, DIFF,36, &earth_tex),//Mirr - SPEC roughtness 0-100
	Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,1,1) * .999, REFR,0, &earth_tex),//Glas
	//Sphere(1.5, Vec(50,81.6 - 16.5,81.6),Vec(4,4,4) * 100,  Vec(), DIFF,0,0),//&light_tex),//Lite																																			
	Sphere(0.5, Vec(27,16.5,20),Vec(4,4,4) * 600,  Vec(), DIFF,0,0),//Lite
};
#endif
int numSpheres = sizeof(spheres) / sizeof(Sphere);
inline double clamp(double x) { return x < 0 ? 0 : x>1 ? 1 : x; }
inline int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }
inline bool intersect(const Ray& r, double& t, int& id) {
	double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
	for (int i = int(n); i--;) if ((d = spheres[i].intersect(r)) && d < t) { t = d; id = i; }
	return t < inf;
}
//
//
typedef struct {
	Vec pos;
	Vec dir;
	Vec lit;
	int id;
	Vec dct;
}pay_load_t;

int light_id = (sizeof(spheres) / sizeof(Sphere) - 1);
//#define _USE_OLD_FAKE_BDRT_
//
Vec light_radiance(std::vector<pay_load_t>& light_path_fifo, Vec init_col_in, const Ray& rx, int depth, unsigned short* Xi, int E = 1, int max_depth = 12) {
	double t;                               // distance to intersection
	int id = 0;                             // id of intersected object
	Vec lit_c = init_col_in;
	Ray r = rx;
	light_path_fifo.clear();
	while (1)
	{
		if (!intersect(r, t, id)) break; // if miss, return black
		const Sphere& obj = spheres[id];        // the hit object
		Vec x = r.o + r.d * t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
		double p = f.x > f.y&& f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
		if (obj.e.x > 0 || obj.e.y > 0 || obj.e.z > 0) { break; }//hit light
		//fix the lit by probability with respect to solid angle
#ifndef _USE_OLD_FAKE_BDRT_
		if (depth == 0 && 0)
		{
			Vec x = r.o + r.d * t;
			Vec sw = spheres[light_id].p - x;
			double cos_a_max = sqrt(1 - spheres[light_id].rad * spheres[light_id].rad / (x - spheres[light_id].p).dot(x - spheres[light_id].p));
			double omega = 2 * M_PI * (1 - cos_a_max);
			printf("");
			lit_c = lit_c * (omega * M_1_PI) * fabs(nl.dot(r.d));
			//
			//lit_c = lit_c * abs(sw.norm().dot(nl));
		}
#endif
		//
#if 0
#else
		if (++depth > 12 || !p)
		{
			if (erand48(Xi) < p && depth <= max_depth)
				f = f * (1 / p);
			else
				//return obj.e * E;
				break;
		}
#endif
		if (obj.refl == DIFF) {                  // Ideal DIFFUSE reflection
												 //BDRT
			f = get_obj_c(obj, x);
			//cosa
			Vec nx = nl.norm();
			Vec dxx = rx.d;
			Vec dx = dxx.norm();
			dx = dx * -1.0;
			double cosa = dx.dot(nx);
			cosa = (cosa < 0.0) ? 0 : cosa;
			//
			pay_load_t px;
			lit_c = lit_c.mult(f);// *(M_1_PI);
			px.lit = lit_c;
			px.pos = x;
			px.dir = r.d;
			px.id = id;
			light_path_fifo.push_back(px);
			//
			double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
			Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
			Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();

			r = (Ray(x, d));
			continue;
		}
		else if (obj.refl == SPEC)              // Ideal SPECULAR reflection
		{
			//return obj.e + f.mult(radiance(Ray(x, r.d - n * 2 * n.dot(r.d)), depth, Xi));
			//
#if 0
			pay_load_t px;
			px.lit = lit_c;
			px.pos = x;
			px.dir = r.d;
			light_path_fifo.push_back(px);
#endif
			r = Ray(x, r.d - n * 2 * n.dot(r.d));
			continue;
			//
		}
		{
#if 0
			pay_load_t px;
			px.lit = lit_c;
			px.pos = x;
			px.dir = r.d;
			light_path_fifo.push_back(px);
#endif
		}
		Ray reflRay(x, r.d - n * 2 * n.dot(r.d));     // Ideal dielectric REFRACTION
		bool into = n.dot(nl) > 0;                // Ray from outside going in?
		double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
		if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0)    // Total internal reflection
		{
			//return obj.e + f.mult(radiance(reflRay, depth, Xi));
			r = reflRay;
			continue;
		}
		Vec tdir = (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
		double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.dot(n));
		double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
		//return obj.e + f.mult(depth > 2 ? (erand48(Xi) < P ?   // Russian roulette
		//    radiance(reflRay, depth, Xi) * RP : radiance(Ray(x, tdir), depth, Xi) * TP) :
		//    radiance(reflRay, depth, Xi) * Re + radiance(Ray(x, tdir), depth, Xi) * Tr);
		//if (depth > 2)
		if (1)
		{
			if (0 && (erand48(Xi) < P))  // Russian roulette - only deal with refraction?
			{
				r = reflRay;
				continue;
			}
			else
			{
				r = Ray(x, tdir);
				continue;
			}
		}
		else
		{
			//radiance(reflRay, depth, Xi)* Re + radiance(Ray(x, tdir), depth, Xi) * Tr)
		}
	}
	return Vec();
}
//
//#define SIMPLE_BDRT
//
double weight_x(double eye_len, double lit_len)
{
	double sum = 0;
	double eye = pow(0.5 * M_1_PI, eye_len);
	for (int i = 0; i < (int)eye_len + (int)lit_len; i++)
	{
		sum += pow(0.5 * M_1_PI, i);
	}
	return eye / sum;
}
Vec radiance(std::vector<pay_load_t>& light_path_fifo, const Ray& rx, int depth, unsigned short* Xi, int E = 1) {
	std::vector<pay_load_t> eye_path_fifo;
	Ray r = rx;
	while (depth < 12)
	{
		double t;                               // distance to intersection
		int id = 0;                               // id of intersected object
		if (!intersect(r, t, id)) return Vec(); // if miss, return black
		const Sphere& obj = spheres[id];        // the hit object
		Vec x = r.o + r.d * t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
#if 1
		f = get_obj_c(obj, x);
		Refl_t mat_property = obj.refl;
		if (mat_property == DIFF && obj.tex)
		{
			//remap mat property for diffuse // spec + diff
			if (f.x < 0.1 && f.y < 0.1 && f.z < 0.1)
			{
				//remap to SPEC
				mat_property = SPEC;
				f = Vec(1, 1, 1);
			}
		}
		if (mat_property == REFR && obj.tex)
		{
			//remap mat property for diffuse // spec + diff
			if (f.x < 0.1 && f.y < 0.1 && f.z < 0.1)
			{
				//remap to SPEC
				mat_property = REFR;
				f = Vec(1, 1, 1);
			}
			else
			{
				mat_property = DIFF;
				//f = Vec(1, 1, 1);
			}
		}
#endif
		double p = f.x > f.y&& f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
#if 0
		if (++depth > 5 || !p) if (erand48(Xi) < p) f = f * (1 / p); else return obj.e * E;
#else
//#define _DIRECT_LIT_
#ifdef _DIRECT_LIT_
		if (++depth > 1 || !p)
		{
			if (erand48(Xi) < p && depth < 5 && 0)
				f = f * (1 / p);
#else
		if (++depth > 5 || !p)
		{
			if (erand48(Xi) < p && depth < 5)
				f = f * (1 / p);
#endif
			else
			{
#ifdef SIMPLE_BDRT
				if (global_en_bdrt == 0)
				{
					return Vec();
				}
				for (int ix = light_path_fifo.size() - 1; ix >= 0; ix--)
				{
					//test visibility
					Ray rrr = Ray(x, (light_path_fifo[ix].pos - x).norm());
					double ttt; int idd;
					if (!intersect(rrr, ttt, idd))
					{
						continue;
					}
					if (idd != light_path_fifo[ix].id) continue;
					//normal -> nl, hit = x
					return light_path_fifo[ix].lit;
				}
#endif
				//return Vec();
				break;
			}
		}
#endif
		Vec gloss_rad;
		if (mat_property == DIFF) {                  // Ideal DIFFUSE reflection
			double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
			Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
			Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();

			// Loop over any lights
			Vec e;
			for (int i = 0; i < numSpheres; i++) {
				const Sphere& s = spheres[i];
				if (s.e.x <= 0 && s.e.y <= 0 && s.e.z <= 0) continue; // skip non-lights

				Vec sw = s.p - x, su = ((fabs(sw.x) > .1 ? Vec(0, 1) : Vec(1)) % sw).norm(), sv = sw % su;
				double cos_a_max = sqrt(1 - s.rad * s.rad / (x - s.p).dot(x - s.p));
				double eps1 = erand48(Xi), eps2 = erand48(Xi);
				double cos_a = 1 - eps1 + eps1 * cos_a_max;
				double sin_a = sqrt(1 - cos_a * cos_a);
				double phi = 2 * M_PI * eps2;
				Vec l = su * cos(phi) * sin_a + sv * sin(phi) * sin_a + sw * cos_a;
				l.norm();
				if (intersect(Ray(x, l), t, id) && id == i) {  // shadow ray
					double omega = 2 * M_PI * (1 - cos_a_max);
					//e = e + f.mult(s.e * l.dot(nl) * omega) * M_1_PI;  // 1/pi for brdf
					e = e + f.mult(get_obj_e(s, x + l*t) * l.dot(nl) * omega) * M_1_PI;
				}
			}
			//continue recursive tracing
			Vec next_level_r;
			//if(en_1)
			pay_load_t ep;
			ep.dct = e + get_obj_e(obj, x) * E; E = 0;
			ep.lit = f;
			ep.dir = d;
			ep.id = id;
			ep.pos = x;
			eye_path_fifo.push_back(ep);
			r = Ray(x, d);
			continue;
			//
			//next_level_r = radiance(light_path_fifo, Ray(x, d), depth, Xi, 0);
			//return get_obj_e(obj, x) * E + e + f.mult(next_level_r);// *(probxxx /* 2 * fabs(nl.dot(d))*/);

		}
		else if (mat_property == SPEC)              // Ideal SPECULAR reflection
		{
			if (obj.gloss)
			{
				//Roughness
				double roughness = (double)obj.gloss / 100.0;
				//
				double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
				//
				r2 = randx(0.0, roughness); r2s = sqrt(r2);
				//
				Vec ref_dir = (r.d - n * 2 * n.dot(r.d)).norm();
				Vec ux = ((fabs(ref_dir.x) > .1 ? Vec(0, 1) : Vec(1)) % ref_dir).norm(), vx = ref_dir % ux;
				Vec d = (ux * cos(r1) * r2s + vx * sin(r1) * r2s + ref_dir * sqrt(1 - r2)).norm();
				double gloss_factor = 1 - r2;
#if 1
				//return obj.e + f.mult(radiance(Ray(x, d), depth, Xi, 0)) /* (obj.gloss ? gloss_factor : 1.0)*/;
				//return get_obj_e(obj, x) + f.mult(radiance(light_path_fifo, Ray(x, d), depth, Xi, 0)) /* (obj.gloss ? gloss_factor : 1.0)*/;
				pay_load_t ep;
				ep.dct = Vec();
				ep.lit = f;
				ep.dir = d;
				ep.id = id;
				ep.pos = x;
				eye_path_fifo.push_back(ep);
				r = Ray(x, d);
				continue;
#else
				//difffactor
				Vec gloss_rad;
				{
					double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
					Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
					Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();

					// Loop over any lights
					Vec e;
					for (int i = 0; i < numSpheres; i++) {
						const Sphere& s = spheres[i];
						if (s.e.x <= 0 && s.e.y <= 0 && s.e.z <= 0) continue; // skip non-lights

						Vec sw = s.p - x, su = ((fabs(sw.x) > .1 ? Vec(0, 1) : Vec(1)) % sw).norm(), sv = sw % su;
						double cos_a_max = sqrt(1 - s.rad * s.rad / (x - s.p).dot(x - s.p));
						double eps1 = erand48(Xi), eps2 = erand48(Xi);
						double cos_a = 1 - eps1 + eps1 * cos_a_max;
						double sin_a = sqrt(1 - cos_a * cos_a);
						double phi = 2 * M_PI * eps2;
						Vec l = su * cos(phi) * sin_a + sv * sin(phi) * sin_a + sw * cos_a;
						l.norm();
						if (intersect(Ray(x, l), t, id) && id == i) {  // shadow ray
							double omega = 2 * M_PI * (1 - cos_a_max);
							e = e + f.mult(s.e * l.dot(nl) * omega) * M_1_PI;  // 1/pi for brdf
						}
					}
					gloss_rad = obj.e * E + e + f.mult(radiance(light_path_fifo, Ray(x, d), depth, Xi, 0));
				}
				return gloss_rad * (obj.gloss ? (1 - gloss_factor) : 0.0) + f.mult(radiance(light_path_fifo, Ray(x, d), depth, Xi, 0)) * (obj.gloss ? gloss_factor : 1.0);
#endif
			}
			else
			{
				//return obj.e + f.mult(radiance(Ray(x, r.d - n * 2 * n.dot(r.d)), depth, Xi));
				//return get_obj_e(obj, x) + f.mult(radiance(light_path_fifo, Ray(x, r.d - n * 2 * n.dot(r.d)), depth, Xi));
				pay_load_t ep;
				ep.dct = Vec();
				ep.lit = f;
				ep.dir = Ray(x, r.d - n * 2 * n.dot(r.d)).d;
				ep.id = id;
				ep.pos = x;
				//eye_path_fifo.push_back(ep);
				r = Ray(x, r.d - n * 2 * n.dot(r.d));
				continue;
			}
		}
		Ray reflRay(x, r.d - n * 2 * n.dot(r.d));     // Ideal dielectric REFRACTION
		bool into = n.dot(nl) > 0;                // Ray from outside going in?
		double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
		if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0)    // Total internal reflection
		{
			//return obj.e + f.mult(radiance(light_path_fifo, reflRay, depth, Xi));
			pay_load_t ep;
			ep.dct = Vec();
			ep.lit = f;
			ep.dir = reflRay.d;
			ep.id = id;
			ep.pos = x;
			//eye_path_fifo.push_back(ep);
			r = reflRay;
			continue;
	    }
		Vec tdir = (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
		double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.dot(n));
		double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
		//return obj.e + f.mult(depth > 2 ? (erand48(Xi) < P ?   // Russian roulette
		///    radiance(reflRay, depth, Xi) * RP : radiance(Ray(x, tdir), depth, Xi) * TP) :
		//    radiance(reflRay, depth, Xi) * Re + radiance(Ray(x, tdir), depth, Xi) * Tr);
#if 0
		return get_obj_e(obj, x) + f.mult(depth > 2 ? (erand48(Xi) < P ?   // Russian roulette
			radiance(light_path_fifo, reflRay, depth, Xi) * RP : radiance(light_path_fifo, Ray(x, tdir), depth, Xi) * TP) :
			radiance(light_path_fifo, reflRay, depth, Xi) * Re + radiance(light_path_fifo, Ray(x, tdir), depth, Xi) * Tr);
#else
		if (0 || (erand48(Xi) < P))
		{
			pay_load_t ep;
			ep.dct = Vec();
			ep.lit = f;
			ep.dir = reflRay.d;
			ep.id = 0;
			ep.id = id;
			ep.pos = x;
			//eye_path_fifo.push_back(ep);
			r = reflRay;
			continue;
		}
		else
		{
			pay_load_t ep;
			ep.dct = Vec();
			ep.lit = f;
			ep.dir = Ray(x, tdir).d;
			ep.id = id;
			ep.pos = x;
			//eye_path_fifo.push_back(ep);
			r = Ray(x, tdir);
			continue;
		}
#endif
	}
	if (eye_path_fifo.size() == 0) return Vec();
	//connect path
#define EYE_AFTER 0
	Vec base_eye = Vec(1,1,1);
	Vec base_dct;
	for (int i = 0; i < 1 && eye_path_fifo.size(); i++)
	{
		//base_dct = base_dct + eye_path_fifo[i].dct;
		//base_eye = base_eye.mult(eye_path_fifo[i].lit);
	}
	Vec total = Vec(0, 0, 0);
	double count = 0;
#if 1
	for (int i = EYE_AFTER; i < eye_path_fifo.size(); i++)
	{
		for (int j = 0; j < light_path_fifo.size(); j++)
		{
			Vec dirx = light_path_fifo[j].pos - eye_path_fifo[i].pos;
			count = count + 1;
			dirx = dirx.norm();
			double ttt;
			int idx;
			int app_pi = 0;
			if (!intersect(Ray(eye_path_fifo[i].pos, dirx), ttt, idx))
			{
				continue;
			}
			else
			{
				if (idx != light_path_fifo[j].id) continue;
				if (light_path_fifo[j].id == eye_path_fifo[i].id)
				{
					//printf("Warning!\n");
					Vec vvv = eye_path_fifo[i].pos - light_path_fifo[j].pos;
					double pow2_dist = vvv.dot(vvv);
					double dist = sqrt(pow2_dist);
					if (dist < 5) {
						app_pi = 0;
						printf("");
					}
					else
						printf("");
					//app_pi = 0;
				}
			}
			//
			Vec tdirx = light_path_fifo[j].pos - eye_path_fifo[i].pos;
			double dist2 = tdirx.dot(tdirx);
			Vec n1 = spheres[light_path_fifo[j].id].p - light_path_fifo[j].pos;
			n1 = n1.norm();
			Vec n2 = spheres[eye_path_fifo[i].id].p - eye_path_fifo[i].pos;
			n2 = n2.norm();
			Vec dir22 = tdirx;
			dir22 = dir22.norm();
			if (dist2 < 1) dist2 = 1;
			//
			Vec t = Vec(1, 1, 1);
			double wx = /*fabs(n1.dot(dir22)) * fabs(n2.dot(dir22)) */ 1.0 / dist2;//weight_x(i + 1, j + 1);
#if 0
			t = t.mult(light_path_fifo[j].lit) * (fabs(n1.dot(dir22)) * fabs(n2.dot(dir22))* 1.0 /* dist2*/);
#else
			t = t.mult(light_path_fifo[j].lit) * wx * (1.0 / (double)(1 + i + j + 1));
#endif
			for (int k = i - (app_pi ? 1 : 0); k >= EYE_AFTER; k--)
			{
				if (k == 0 && 0)
				{
					t = t.mult(eye_path_fifo[k].lit) +eye_path_fifo[k].dct;
				}
				else
				{
					t = t.mult(eye_path_fifo[k].lit) + eye_path_fifo[k].dct;
				}
			}
			total = total + t;
		}
	}
#else
	int e_i = eye_path_fifo.size() - 1;
	int l_i = light_path_fifo.size() - 1;
	while (e_i >= 0 && l_i >= 0)
	{
		Vec dirx = light_path_fifo[l_i].pos - eye_path_fifo[e_i].pos;
		count = count + 1;
		dirx = dirx.norm();
		double ttt;
		int idx;
		int app_pi = 0;
		if (!intersect(Ray(eye_path_fifo[e_i].pos, dirx), ttt, idx))
		{
			e_i--;
			l_i--;
			continue;
		}
		else
		{
			if (idx != light_path_fifo[l_i].id)
			{
				e_i--;
				l_i--;
				continue;
			}
			if (light_path_fifo[l_i].id == eye_path_fifo[e_i].id)
			{
				//printf("Warning!\n");
				Vec vvv = eye_path_fifo[e_i].pos - light_path_fifo[l_i].pos;
				double pow2_dist = vvv.dot(vvv);
				double dist = sqrt(pow2_dist);
				if (dist < 1)
				{
					e_i--;
					l_i--;
					continue;
				}
				else
				{
					e_i--;
					l_i--;
					continue;
				}
				//app_pi = 0;
			}
		}
		//
		Vec t = Vec(1, 1, 1);
		t = t.mult(light_path_fifo[l_i].lit);
		for (int x = e_i; x >= 0; x--)
		{
			t = t.mult(eye_path_fifo[x].lit) + eye_path_fifo[x].dct;
		}
		total = total + t * (1.0 / (double)(e_i + l_i +1));
		//
		e_i--;
		l_i--;
	}
#endif
	total = total + eye_path_fifo[0].dct;
	total = total * (1.0 / (double)(count+1));
	return total;
}

Vec plane_intersect(Vec plane_norm, Vec plane_p, Vec ray_o, Vec ray_dir)
{
	//n_plane * (p - p_plane) = 0
	//n_plane * (ray_o + ray_dir * t - p_plane) = 0
	//n_plane * ray_o + n_plane * ray_dir * t = n_plane * p_plane
	//t = (n_plane * (p_plane - ray_o)) / (n_plane* ray_dir)
	double t = plane_norm.dot(plane_p - ray_o) / (plane_norm.dot(ray_dir));
	return ray_o + ray_dir * t;
}
#define SHOW_OPENGL
//http://sourceware.org/pthreads-win32/
//https://my.oschina.net/u/2245781/blog/3023665
//nuget nupengl
#ifdef SHOW_OPENGL
#include <windows.h>
#include <gl/glut.h>
int g_argc;
char** g_argv;
void renderWindow2();
int rotate = 0;
Vec* g_c = 0;
void renderWindow2(void)
{
#ifdef SHOW_OPENGL
	glColor3f(1.0, rotate ? 1.0 : 0.0, 0.0);
	rotate = rotate ? 0 : 1;
	int w = 1024, h = 768;
	for (int y = 0; y<h; y++)
		for (int x = 0; x < w; x++)
		{
			int it = (h - y - 1) * w + x;
			glColor3f(toInt(g_c[it].x) / 255.0, toInt(g_c[it].y) / 255.0, toInt(g_c[it].z) / 255.0);
			glBegin(GL_POINTS);
			glVertex2i(x, h - y - 1);
			glEnd();
		}
	glFlush();
#endif
}
void glutTimer(int value)
{
	glutPostRedisplay();
	glutTimerFunc(1000, glutTimer, 1);
}
DWORD WINAPI open_new_thread_ogl2(PVOID pvParam);
void open_new_thread_ogl(int argc, char* argv[])
{
	g_argc = argc;
	g_argv = argv;
	HANDLE hThread1 = CreateThread(NULL, 0, open_new_thread_ogl2, NULL, 0, NULL);
	//CloseHandle(hThread1);
}
//DWORD WINAPI MyThread2(PVOID pvParam)
DWORD WINAPI open_new_thread_ogl2(PVOID pvParam)
{
#ifdef SHOW_OPENGL
#ifdef _TEST_OPENGL
#else
	glutInit(&g_argc, g_argv);                                  //初始化glut: 接收主函数的参数
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);            //显示模式：颜色&缓冲
	glutInitWindowPosition(0, 0);                           //窗口相对屏幕位置
	glutInitWindowSize(800, 600);                           //窗口大小
	glutCreateWindow("Hello, OpenGL!");                     //创建窗口: 标题
	glClear(GL_COLOR_BUFFER_BIT);                           //当前背景色填充窗口
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, 1024.0, 768.0, 0.0);
	glutDisplayFunc(&renderWindow2);
	glutTimerFunc(1000, glutTimer, 1);
	glutMainLoop();                                         //循环
															//return 0;
#endif
#endif
	return 0;
}
#endif

Vec radiance_org(std::vector<pay_load_t>& light_path_fifo, const Ray& r, int depth, unsigned short* Xi, int E = 1) {
	double t;                               // distance to intersection
	int id = 0;                               // id of intersected object
	if (!intersect(r, t, id)) return Vec(); // if miss, return black
	const Sphere& obj = spheres[id];        // the hit object
	Vec x = r.o + r.d * t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
#if 1
	f = get_obj_c(obj, x);
	Refl_t mat_property = obj.refl;
	if (mat_property == DIFF && obj.tex)
	{
		//remap mat property for diffuse // spec + diff
		if (f.x < 0.1 && f.y < 0.1 && f.z < 0.1)
		{
			//remap to SPEC
			mat_property = SPEC;
			f = Vec(1, 1, 1);
		}
	}
	if (mat_property == REFR && obj.tex)
	{
		//remap mat property for diffuse // spec + diff
		if (f.x < 0.1 && f.y < 0.1 && f.z < 0.1)
		{
			//remap to SPEC
			mat_property = REFR;
			f = Vec(1, 1, 1);
		}
		else
		{
			mat_property = DIFF;
			//f = Vec(1, 1, 1);
		}
	}
#endif
	double p = f.x > f.y&& f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
#if 0
	if (++depth > 5 || !p) if (erand48(Xi) < p) f = f * (1 / p); else return obj.e * E;
#else
	if (++depth > 5 || !p)
	{
		if (erand48(Xi) < p && depth < 5)
			f = f * (1 / p);
		else
			return obj.e * E;
	}
#endif
	Vec gloss_rad;
	if (mat_property == DIFF) {                  // Ideal DIFFUSE reflection
		double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
		Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
		Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();

		// Loop over any lights
		Vec e;
		for (int i = 0; i < numSpheres; i++) {
			const Sphere& s = spheres[i];
			if (s.e.x <= 0 && s.e.y <= 0 && s.e.z <= 0) continue; // skip non-lights

			Vec sw = s.p - x, su = ((fabs(sw.x) > .1 ? Vec(0, 1) : Vec(1)) % sw).norm(), sv = sw % su;
			double cos_a_max = sqrt(1 - s.rad * s.rad / (x - s.p).dot(x - s.p));
			double eps1 = erand48(Xi), eps2 = erand48(Xi);
			double cos_a = 1 - eps1 + eps1 * cos_a_max;
			double sin_a = sqrt(1 - cos_a * cos_a);
			double phi = 2 * M_PI * eps2;
			Vec l = su * cos(phi) * sin_a + sv * sin(phi) * sin_a + sw * cos_a;
			l.norm();
			if (intersect(Ray(x, l), t, id) && id == i) {  // shadow ray
				double omega = 2 * M_PI * (1 - cos_a_max);
				//e = e + f.mult(s.e * l.dot(nl) * omega) * M_1_PI;  // 1/pi for brdf
				e = e + f.mult(get_obj_e(s, x + l*t) * l.dot(nl) * omega) * M_1_PI;
			}
		}
#if 0
		//return obj.e * E + e + f.mult(radiance(Ray(x, d), depth, Xi, 0));
		return get_obj_e(obj, x) * E + e + f.mult(radiance(light_path_fifo, Ray(x, d), depth, Xi, 0));
#else
		Vec BDRT_e;
		//calculate indirect illumilation from light path
		for (int ix = 0; ix < light_path_fifo.size(); ix++)
		{
			//test visibility
			Ray rrr = Ray(x, (light_path_fifo[ix].pos - x).norm());
			double ttt; int idd;
			if (!intersect(rrr, ttt, idd))
			{
				continue;
			}
			if (idd != light_path_fifo[ix].id) continue;
			//normal -> nl, hit = x
			Vec rxx = r.d;
			double  cos_theta = nl.norm().dot(rxx.norm());
			if (cos_theta < 0) cos_theta = -cos_theta;
#if 0
			BDRT_e = BDRT_e + light_path_fifo[ix].lit.mult(f) * cos_theta * (0.008 / (light_path_fifo.size() * M_PI * M_PI));
#else
			Vec vvv2 = x - r.o;
			BDRT_e = BDRT_e + light_path_fifo[ix].lit.mult(f) * cos_theta * (1.0 / (light_path_fifo.size()));
#endif
		}
		//continue recursive tracing
#if 0
		return get_obj_e(obj, x) * E + e + BDRT_e + f.mult(radiance(light_path_fifo, Ray(x, d), depth, Xi, 0));
#else
		double probxxx = (light_path_fifo.size() || global_en_bdrt) ? 0.5 : 0.0;
		return get_obj_e(obj, x) * E + e + BDRT_e * probxxx + f.mult(radiance_org(light_path_fifo, Ray(x, d), depth, Xi, 0)) * (1 - probxxx);
#endif
#endif
	}
	else if (mat_property == SPEC)              // Ideal SPECULAR reflection
	{
		if (obj.gloss)
		{
			//Roughness
			double roughness = (double)obj.gloss / 100.0;
			//
			double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
			//
			r2 = randx(0.0, roughness); r2s = sqrt(r2);
			//
			Vec ref_dir = (r.d - n * 2 * n.dot(r.d)).norm();
			Vec ux = ((fabs(ref_dir.x) > .1 ? Vec(0, 1) : Vec(1)) % ref_dir).norm(), vx = ref_dir % ux;
			Vec d = (ux * cos(r1) * r2s + vx * sin(r1) * r2s + ref_dir * sqrt(1 - r2)).norm();
			double gloss_factor = 1 - r2;
#if 1
			//return obj.e + f.mult(radiance(Ray(x, d), depth, Xi, 0)) /* (obj.gloss ? gloss_factor : 1.0)*/;
			return get_obj_e(obj, x) + f.mult(radiance_org(light_path_fifo, Ray(x, d), depth, Xi, 0)) /* (obj.gloss ? gloss_factor : 1.0)*/;
#else
			//difffactor
			Vec gloss_rad;
			{
				double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
				Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
				Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();

				// Loop over any lights
				Vec e;
				for (int i = 0; i < numSpheres; i++) {
					const Sphere& s = spheres[i];
					if (s.e.x <= 0 && s.e.y <= 0 && s.e.z <= 0) continue; // skip non-lights

					Vec sw = s.p - x, su = ((fabs(sw.x) > .1 ? Vec(0, 1) : Vec(1)) % sw).norm(), sv = sw % su;
					double cos_a_max = sqrt(1 - s.rad * s.rad / (x - s.p).dot(x - s.p));
					double eps1 = erand48(Xi), eps2 = erand48(Xi);
					double cos_a = 1 - eps1 + eps1 * cos_a_max;
					double sin_a = sqrt(1 - cos_a * cos_a);
					double phi = 2 * M_PI * eps2;
					Vec l = su * cos(phi) * sin_a + sv * sin(phi) * sin_a + sw * cos_a;
					l.norm();
					if (intersect(Ray(x, l), t, id) && id == i) {  // shadow ray
						double omega = 2 * M_PI * (1 - cos_a_max);
						e = e + f.mult(s.e * l.dot(nl) * omega) * M_1_PI;  // 1/pi for brdf
					}
				}
				gloss_rad = obj.e * E + e + f.mult(radiance(light_path_fifo, Ray(x, d), depth, Xi, 0));
			}
			return gloss_rad * (obj.gloss ? (1 - gloss_factor) : 0.0) + f.mult(radiance(light_path_fifo, Ray(x, d), depth, Xi, 0)) * (obj.gloss ? gloss_factor : 1.0);
#endif
		}
		else
		{
			//return obj.e + f.mult(radiance(Ray(x, r.d - n * 2 * n.dot(r.d)), depth, Xi));
			return get_obj_e(obj, x) + f.mult(radiance_org(light_path_fifo, Ray(x, r.d - n * 2 * n.dot(r.d)), depth, Xi));
		}
	}
	Ray reflRay(x, r.d - n * 2 * n.dot(r.d));     // Ideal dielectric REFRACTION
	bool into = n.dot(nl) > 0;                // Ray from outside going in?
	double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
	if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0)    // Total internal reflection
		return obj.e + f.mult(radiance_org(light_path_fifo, reflRay, depth, Xi));
	Vec tdir = (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
	double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.dot(n));
	double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
	//return obj.e + f.mult(depth > 2 ? (erand48(Xi) < P ?   // Russian roulette
	///    radiance(reflRay, depth, Xi) * RP : radiance(Ray(x, tdir), depth, Xi) * TP) :
	//    radiance(reflRay, depth, Xi) * Re + radiance(Ray(x, tdir), depth, Xi) * Tr);
	return get_obj_e(obj, x) + f.mult(depth > 2 ? (erand48(Xi) < P ?   // Russian roulette
		radiance_org(light_path_fifo, reflRay, depth, Xi) * RP : radiance_org(light_path_fifo, Ray(x, tdir), depth, Xi) * TP) :
		radiance_org(light_path_fifo, reflRay, depth, Xi) * Re + radiance_org(light_path_fifo, Ray(x, tdir), depth, Xi) * Tr);
}

int main(int argc, char* argv[]) {
	//
	open_new_thread_ogl(argc, argv);
	//
	int w = 1024, h = 768, samps = argc >= 2 ? atoi(argv[1]) / 4 : 1; // # samples
	Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // cam pos, dir
	Vec cx = Vec(w * .5135 / h), cy = (cx % cam.d).norm() * .5135, r, *c = new Vec[w * h];
#ifdef SHOW_OPENGL
	g_c = c;
#endif
	int en_bdrt = (argc >= 6) ? atoi(argv[5]) : 0;
	global_en_bdrt = en_bdrt;
	if (!en_bdrt)
	{
#if 0
		spheres[light_id].e = Vec(4, 4, 4) * 118;//adjust brightness since BDRT will be brighter
#endif
	}
	//
	light_tex.load_image(argv[2]);
	earth_tex.load_image(argv[3]);
	int dof_level = 0;
	if (argc >= 5)
	{
		dof_level = atoi(argv[4]);
	}
	int dof_aperture = 2 * dof_level + 1;//1,3,5,7,9,11,13,15,17.... 2*n+1
										 //
#pragma omp parallel for schedule(dynamic, 1) private(r)      // OpenMP
	for (int y = 0; y < h; y++) {                       // Loop over image rows
														//if (y > 10) break;//test baise
		fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100. * y / (h - 1));
		for (unsigned short x = 0, Xi[3] = { 0,0,y * y * y }; x < w; x++) {   // Loop cols
																			  //DOF here
																			  //calculate the image plane and focal plane
																			  //normal using gaze dir
			Vec gaze_dir = cam.d;
			Vec image_plane = cam.o + cam.d * 200;
			Vec focal_plane = cam.o + cam.d * 210;
			//
#ifdef _NO_DOF_
			for (int sy = 0, i = (h - y - 1) * w + x; sy < 2; sy++)     // 2x2 subpixel rows
				for (int sx = 0; sx < 2; sx++, r = Vec()) {        // 2x2 subpixel cols
					for (int s = 0; s < samps; s++) {
						double r1 = 2 * erand48(Xi), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
						double r2 = 2 * erand48(Xi), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
						Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
							cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
						r = r + radiance(Ray(cam.o + d * 140, d.norm()), 0, Xi) * (1. / samps);
					} // Camera rays are pushed ^^^^^ forward to start in interior
					c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
				}
#else
			//
			//if (x != 421 || y != 168) continue;
			for (int sy = 0, i = (h - y - 1) * w + x; sy < 2; sy++)     // 2x2 subpixel rows
				for (int sx = 0; sx < 2; sx++, r = Vec()) {        // 2x2 subpixel cols

					for (int dof_x = -(dof_aperture / 2); dof_x < (dof_aperture / 2 + 1); dof_x++)
						for (int dof_y = -(dof_aperture / 2); dof_y < (dof_aperture / 2 + 1); dof_y++)
						{
							for (int s = 0; s < samps; s++) {
								double r1 = 2 * erand48(Xi), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
								double r2 = 2 * erand48(Xi), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
								Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
									cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
								//
								Vec focal_plane_hit = plane_intersect(gaze_dir, focal_plane, cam.o, d);
								//for (int dof_x = -1; dof_x < 2; dof_x++)
								{
									//for (int dof_y = -1; dof_y < 2; dof_y++)
									{
										if (x + dof_x < 0) dof_x = 0;
										if (y + dof_y < 0) dof_y = 0;
										Vec d2 = cx * (((sx + .5 + dx) / 2 + x + dof_x) / w - .5) +
											cy * (((sy + .5 + dy) / 2 + y + dof_y) / h - .5) + cam.d;
										Vec image_plane_hit = plane_intersect(gaze_dir, image_plane, cam.o, d2);
										Vec new_dir = (focal_plane_hit - image_plane_hit).norm();
										//
										//Trace light
										std::vector<pay_load_t> light_path_fifo;
										unsigned short us = 0;
										if (en_bdrt)
										{
#define _LIT_SAMPLE_FACTOR_ (0.0013 * M_1_PI) //use the random sample tech in radiance()'s DIFF (Direct light sampling) later
											double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
											Vec w = Vec(0, 1, 0), u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
											Vec lit_d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
											//
											Vec dir((erand48(Xi) - 0.5) * 2, (erand48(Xi) - 0.5) * 2, (erand48(Xi) - 0.5) * 2);
											lit_d = dir.norm();
											//
#ifdef _USE_OLD_FAKE_BDRT_
											light_radiance(light_path_fifo, spheres[light_id].e * (_LIT_SAMPLE_FACTOR_), Ray(spheres[light_id].p + lit_d * (spheres[light_id].rad + 0.1), lit_d), 0, &us);
#else
											light_radiance(light_path_fifo, spheres[light_id].e/* * (_LIT_SAMPLE_FACTOR_)*/, Ray(spheres[light_id].p + lit_d * (spheres[light_id].rad + 0.1), lit_d), 0, &us);
#endif
										}
										//
										if (!en_bdrt)
										{
											Vec radx = radiance_org(light_path_fifo, Ray(image_plane_hit + new_dir * -60, new_dir.norm()), 0, Xi);
											r = r + radx * (1. / samps);
											//printf("\n%.3f %.3f %.3f\n", radx.x, radx.y, radx.z);
										}
										else
										{
											Vec radx = radiance(light_path_fifo, Ray(image_plane_hit + new_dir * -60, new_dir.norm()), 0, Xi);
											r = r + radx * (1. / samps);
											//printf("\n%.3f %.3f %.3f\n", radx.x, radx.y, radx.z);
										}
									}
								}
							}
						} // Camera rays are pushed ^^^^^ forward to start in interior
					r = r * (1. / (double)(dof_aperture * dof_aperture));
					c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
#endif
				}
		}
	}
	FILE* f = fopen("image.ppm", "w");         // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for (int i = 0; i < w * h; i++)
		fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}
