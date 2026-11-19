// Small CPU kernels for triangle/cell clipping and point-to-boundary queries.
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

using Point = std::array<double, 3>;
using Polygon = std::array<Point, 12>;

// Clip a convex polygon against one vertical side of the grid cell.
static int clip(const Polygon& in, int n, Polygon& out, int axis, double at, int sign) {
    int count = 0;
    for (int k=0; k<n; ++k) {
        Point a=in[(k+n-1)%n], b=in[k];
        double da=(a[axis]-at)*sign, db=(b[axis]-at)*sign;
        if ((da<0) != (db<0)) {
            double t=da/(da-db);
            Point p;
            for (int d=0; d<3; ++d) p[d]=a[d]+t*(b[d]-a[d]);
            out[count++]=p;
        }
        if (db>=0) out[count++]=b;
    }
    return count;
}

extern "C" void raster(const double* verts, const int* faces, int nf, int nx, int ny,
                        double ox, double oy, double res, float* surface, float* atlas,
                        float* top, unsigned char* soft, int mode, const double* bands) {
    // mode 0 samples the top surface at a point; modes 1/2 accumulate visible
    // triangle fragments over the full cell (mode 2 also records foliage).
    for (int f=0; f<nf; ++f) {
        Polygon tri;
        for (int k=0; k<3; ++k) {
            auto v=verts+3*faces[3*f+k];
            tri[k]={(v[0]-ox)/res,(v[1]-oy)/res,v[2]};
        }
        auto a=tri[0], b=tri[1], c=tri[2];
        double det=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
        if (!mode && std::abs(det)<1e-14) continue;
        double margin=mode ? .5 : 1e-8;
        int x0=std::max(0,(int)std::ceil(std::min({a[0],b[0],c[0]})-margin));
        int x1=std::min(nx-1,(int)std::floor(std::max({a[0],b[0],c[0]})+margin));
        int y0=std::max(0,(int)std::ceil(std::min({a[1],b[1],c[1]})-margin));
        int y1=std::min(ny-1,(int)std::floor(std::max({a[1],b[1],c[1]})+margin));
        for (int i=x0;i<=x1;++i) for (int j=y0;j<=y1;++j) {
            int idx=i*ny+j;
            if (!mode) {
                double u=((i-a[0])*(c[1]-a[1])-(j-a[1])*(c[0]-a[0]))/det;
                double v=((b[0]-a[0])*(j-a[1])-(b[1]-a[1])*(i-a[0]))/det;
                if (std::min({u,v,1-u-v})>=-1e-8)
                    surface[idx]=std::max(surface[idx],float(a[2]+u*(b[2]-a[2])+v*(c[2]-a[2])));
                continue;
            }
            Polygon p,q;
            int n=clip(tri,3,p,0,i-.5,1);
            n=clip(p,n,q,0,i+.5,-1);
            n=clip(q,n,p,1,j-.5,1);
            n=clip(p,n,q,1,j+.5,-1);
            if (!n) continue;
            double lo=q[0][2], hi=lo;
            for (int k=1;k<n;++k) { lo=std::min(lo,q[k][2]); hi=std::max(hi,q[k][2]); }
            top[idx]=std::max(top[idx],float(hi));
            if (mode==2) soft[idx]=1;
            lo-=surface[idx]; hi-=surface[idx];
            float* cell=atlas+20*idx;
            for (int k=0;k<6;++k) {
                double lower=bands[2*k], upper=bands[2*k+1];
                if (hi<lower || lo>=upper) continue;
                float* band=cell+1+3*k;
                float bot=std::max(lo,lower);
                band[0]=band[2] ? std::min(band[0],bot) : bot;
                band[1]=std::max(band[1],std::min(float(hi),std::nextafter(float(upper),float(lower))));
                band[2]=1;
            }
        }
    }
}

struct Segment { double x,y,u,v; };
struct Node { double x0,y0,x1,y1; int begin,end,left=-1,right=-1; };

struct Boundary {
    std::vector<Segment> edges;
    std::vector<Node> nodes;
    int build(int begin, int end) {
        Node n={INFINITY,INFINITY,-INFINITY,-INFINITY,begin,end};
        for (int i=begin;i<end;++i) {
            auto s=edges[i];
            n.x0=std::min({n.x0,s.x,s.u}); n.x1=std::max({n.x1,s.x,s.u});
            n.y0=std::min({n.y0,s.y,s.v}); n.y1=std::max({n.y1,s.y,s.v});
        }
        int id=nodes.size();
        nodes.push_back(n);
        if (end-begin>8) {
            bool x=n.x1-n.x0>n.y1-n.y0;
            int mid=(begin+end)/2;
            std::nth_element(edges.begin()+begin,edges.begin()+mid,edges.begin()+end,
                [x](auto a,auto b){return x ? a.x+a.u<b.x+b.u : a.y+a.v<b.y+b.v;});
            int left=build(begin,mid), right=build(mid,end);
            nodes[id].left=left; nodes[id].right=right;
        }
        return id;
    }
    double bound(int id,double x,double y) const {
        auto &n=nodes[id];
        double dx=std::max({n.x0-x,x-n.x1,0.}),dy=std::max({n.y0-y,y-n.y1,0.});
        return dx*dx+dy*dy;
    }
    void nearest(int id,double x,double y,double& best) const {
        auto &n=nodes[id];
        if (n.left<0) {
            for (int i=n.begin;i<n.end;++i) {
                auto s=edges[i];
                double dx=s.u-s.x,dy=s.v-s.y,px=x-s.x,py=y-s.y,len=dx*dx+dy*dy;
                double t=len ? std::clamp((px*dx+py*dy)/len,0.,1.) : 0;
                px-=t*dx; py-=t*dy;
                best=std::min(best,px*px+py*py);
            }
        } else {
            int a=n.left,b=n.right;
            double da=bound(a,x,y),db=bound(b,x,y);
            if (db<da) {std::swap(a,b); std::swap(da,db);}
            if (da<best) nearest(a,x,y,best);
            if (db<best) nearest(b,x,y,best);
        }
    }
};

extern "C" void distances(const double* segments,int count,int nx,int ny,double ox,double oy,
                          double res,float* out) {
    Boundary tree;
    for (int i=0;i<count;++i) {
        auto s=segments+i*4;
        tree.edges.push_back({s[0],s[1],s[2],s[3]});
    }
    tree.build(0,count);
    #pragma omp parallel for schedule(static)
    for (int i=0;i<nx;++i) for (int j=0;j<ny;++j) {
        double d=std::numeric_limits<double>::infinity();
        tree.nearest(0,ox+i*res,oy+j*res,d);
        out[i*ny+j]=std::sqrt(d);
    }
}

