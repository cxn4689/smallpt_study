// SmallPt_Original.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>

#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2009
#include <stdlib.h> // Make : g++ -O3 -fopenmp explicit.cpp -o explicit
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include "base.h"
#include <vector>
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
struct Ray { Vec o, d; Ray(Vec o_, Vec d_) : o(o_), d(d_) {} };
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()
struct Sphere {
    double rad;       // radius
    Vec p, e, c;      // position, emission, color
    Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
    Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) :
        rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
    double intersect(const Ray& r) const { // returns distance, 0 if nohit
        Vec op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t, eps = 1e-4, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
        if (det < 0) return 0; else det = sqrt(det);
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    }
};
#if 0
Sphere spheres[] = {//Scene: radius, position, emission, color, material
  Sphere(1e5, Vec(1e5 + 1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF),//Left0
  Sphere(1e5, Vec(-1e5 + 99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF),//Rght1
  Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.75,.75),DIFF),//Back2
  Sphere(1e5, Vec(50,40.8,-1e5 + 170), Vec(),Vec(),           DIFF),//Frnt3
  Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF),//Botm4
  Sphere(1e5, Vec(50,-1e5 + 81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top5
  Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(1,1,1) * .999, SPEC),//Mirr6
  Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,1,1) * .999, REFR),//Glas7
  Sphere(1.5, Vec(50,81.6 - 16.5,81.6),Vec(4,4,4) * 100,  Vec(), DIFF),//Lite8
};
#else
Sphere spheres[] = {//Scene: radius, position, emission, color, material
  Sphere(1e5, Vec(1e5 + 1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF),//Left0
  Sphere(1e5, Vec(-1e5 + 99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF),//Rght1
  Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.75,.75),DIFF),//Back2
  Sphere(1e5, Vec(50,40.8,-1e5 + 170), Vec(),Vec(),           DIFF),//Frnt3
  Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF),//Botm4
  Sphere(1e5, Vec(50,-1e5 + 81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top5
  Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(1,1,1) * .999, SPEC),//Mirr6
  Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,1,1) * .999, REFR),//Glas7
  Sphere(1.5, Vec(27,16.5,20),Vec(4,4,4) * 100,  Vec(), DIFF),//Lite8
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
typedef struct {
    Vec pos;
    Vec dir;
    Vec lit;
    int id;
}pay_load_t;

int light_id = (sizeof(spheres) / sizeof(Sphere)-1);
//
Vec light_radiance(std::vector<pay_load_t> & light_path_fifo, Vec init_col, const Ray& rx, int depth, unsigned short* Xi, int E = 1, int max_depth=5) {
    double t;                               // distance to intersection
    int id = 0;                             // id of intersected object
    Vec lit_c = init_col;
    Ray r = rx;
    light_path_fifo.clear();
    while (1)
    {
        if (!intersect(r, t, id)) break; // if miss, return black
        const Sphere& obj = spheres[id];        // the hit object
        Vec x = r.o + r.d * t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
        double p = f.x > f.y&& f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
        if (obj.e.x > 0 || obj.e.y > 0 || obj.e.z > 0) { break; }//hit light
#if 0
#else
        if (++depth > 5 || !p)
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
            lit_c = lit_c.mult(f);
            pay_load_t px;
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
        if(1)
        {
            if ((erand48(Xi) < P))  // Russian roulette
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
Vec radiance(std::vector<pay_load_t>& light_path_fifo, const Ray& r, int depth, unsigned short* Xi, int E = 1) {
    double t;                               // distance to intersection
    int id = 0;                               // id of intersected object
    if (!intersect(r, t, id)) return Vec(); // if miss, return black
    const Sphere& obj = spheres[id];        // the hit object
    Vec x = r.o + r.d * t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
    double p = f.x > f.y&& f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
#if 0
    if (++depth > 5 || !p) if (erand48(Xi) < p) f = f * (1 / p); else return obj.e * E;
#else
    if (++depth > 5 || !p)
    {
        if (erand48(Xi) < p && depth <= 35)
            f = f * (1 / p);
        else
            return obj.e * E;
    }
#endif
    if (obj.refl == DIFF) {                  // Ideal DIFFUSE reflection
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
#if 0
        return obj.e * E + e + f.mult(radiance(Ray(x, d), depth, Xi, 0));
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
            BDRT_e = BDRT_e + light_path_fifo[ix].lit.mult(f) * cos_theta * (0.008 / (light_path_fifo.size() * M_PI*M_PI));
        }
        //continue recursive tracing
        return obj.e * E + e + BDRT_e + f.mult(radiance(light_path_fifo, Ray(x, d), depth, Xi, 0));
#endif
    }
    else if (obj.refl == SPEC)              // Ideal SPECULAR reflection
        return obj.e + f.mult(radiance(light_path_fifo, Ray(x, r.d - n * 2 * n.dot(r.d)), depth, Xi));
    Ray reflRay(x, r.d - n * 2 * n.dot(r.d));     // Ideal dielectric REFRACTION
    bool into = n.dot(nl) > 0;                // Ray from outside going in?
    double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
    if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0)    // Total internal reflection
        return obj.e + f.mult(radiance(light_path_fifo, reflRay, depth, Xi));
    Vec tdir = (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
    double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.dot(n));
    double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
    return obj.e + f.mult(depth > 2 ? (erand48(Xi) < P ?   // Russian roulette
        radiance(light_path_fifo, reflRay, depth, Xi) * RP : radiance(light_path_fifo, Ray(x, tdir), depth, Xi) * TP) :
        radiance(light_path_fifo, reflRay, depth, Xi) * Re + radiance(light_path_fifo, Ray(x, tdir), depth, Xi) * Tr);
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
    for (int y = 0; y < h; y++)
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

int main(int argc, char* argv[]) {
    //
#ifdef SHOW_OPENGL
    open_new_thread_ogl(argc, argv);
#endif
    //
    //Test
    unsigned short us;
    //light_radiance(spheres[light_id].e, Ray(spheres[light_id].p+ Vec(0.1, -0.8, 0.59)*2, Vec(0.1, -0.8, 0.59)), 0, &us);
    //
    int w = 1024, h = 768, samps = argc >= 2 ? atoi(argv[1]) / 4 : 1; // # samples
    Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // cam pos, dir
    Vec cx = Vec(w * .5135 / h), cy = (cx % cam.d).norm() * .5135, r, * c = new Vec[w * h];
    //
    int en_bdrt = (argc > 2) ? atoi(argv[2]) : 0;
    if (!en_bdrt)
    {
        spheres[light_id].e = Vec(4, 4, 4) * 118;//adjust brightness since BDRT will be brighter
    }
#ifdef SHOW_OPENGL
    g_c = c;
#endif
#pragma omp parallel for schedule(dynamic, 1) private(r)       // OpenMP
    for (int y = 0; y < h; y++) {                       // Loop over image rows
        fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100. * y / (h - 1));
        for (unsigned short x = 0, Xi[3] = { 0,0,y * y * y }; x < w; x++)   // Loop cols
            for (int sy = 0, i = (h - y - 1) * w + x; sy < 2; sy++)     // 2x2 subpixel rows
                for (int sx = 0; sx < 2; sx++, r = Vec()) {        // 2x2 subpixel cols
                    for (int s = 0; s < samps; s++) {
                        double r1 = 2 * erand48(Xi), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                        double r2 = 2 * erand48(Xi), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                        Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                            cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
                        //Trace light
                        std::vector<pay_load_t> light_path_fifo;
                        if(en_bdrt)
                        {
                            double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
                            Vec w = Vec(0,1,0), u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
                            Vec lit_d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
                            light_radiance(light_path_fifo, spheres[light_id].e, Ray(spheres[light_id].p + lit_d * (spheres[light_id].rad+0.1), lit_d), 0, &us);
                        }
                        //
                        r = r + radiance(light_path_fifo, Ray(cam.o + d * 140, d.norm()), 0, Xi) * (1. / samps);
                    } // Camera rays are pushed ^^^^^ forward to start in interior
                    c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
                }
    }
    FILE* f = fopen("image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w * h; i++)
        fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}
