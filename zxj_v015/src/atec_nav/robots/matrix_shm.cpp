// Simulation adapter for the Robot MC v0.6.6 shared-memory ABI.
// Field layouts were checked against the release's ELF debug type information.
// This is project-owned interoperability code, not controller/policy source.
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>

namespace bip = boost::interprocess;
struct StampName { uint64_t timestamp_ms; char name[64]; };
struct JointCmd : StampName { double pos, vel, tor, kp, kd; };
struct JointState : StampName { double pos, vel, tor, temp, voltage; bool enable; };
struct Imu : StampName { double acc[3], gyro[3], quat[4]; };
struct InitControl {
    bip::interprocess_mutex mtx;
    bip::interprocess_condition cond;
    std::atomic<bool> ready{false}, failed{false};
    std::atomic<int> refs{1};
};
static_assert(sizeof(JointCmd) == 112 && sizeof(JointState) == 120);
static_assert(sizeof(Imu) == 152 && sizeof(InitControl) == 96);

template<class T> class Segment {
public:
    using Alloc = bip::allocator<T, bip::managed_shared_memory::segment_manager>;
    using Vec = bip::vector<T, Alloc>;
    std::string name;
    bip::managed_shared_memory memory;
    bip::interprocess_mutex *mutex;
    bip::interprocess_condition *condition;
    Vec *vec;
    Segment(const char *n, int count): name(n), memory(bip::create_only,n,1048576) {
        auto *init = memory.construct<InitControl>("init_control")();
        mutex = memory.construct<bip::interprocess_mutex>("VectorMutex")();
        condition = memory.construct<bip::interprocess_condition>("VectorCond")();
        vec = memory.construct<Vec>("SharedVector")(Alloc(memory.get_segment_manager()));
        vec->reserve(128); vec->resize(count);
        init->ready.store(true);
    }
    ~Segment() { bip::shared_memory_object::remove(name.c_str()); }
};

static uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Bridge {
    Segment<JointCmd> cmd{"joint_cmd",16};
    Segment<JointState> state{"joint_state",16};
    Segment<Imu> imu{"imu_central",1};
    Bridge() {
        const char *legs[] = {"fr","fl","br","bl"};
        const char *parts[] = {"1_hip_roll","2_hip_pitch","3_knee_pitch","4_foot"};
        for(int l=0;l<4;++l) for(int j=0;j<4;++j) {
            auto name=std::string(legs[l])+parts[j];
            std::strcpy((*cmd.vec)[4*l+j].name,name.c_str());
            std::strcpy((*state.vec)[4*l+j].name,name.c_str());
        }
        std::strcpy((*imu.vec)[0].name,"imu_central");
    }
};
static thread_local std::string last_error;
extern "C" {
const char *atec_matrix_error() { return last_error.c_str(); }
void *atec_matrix_create() {
    try { return new Bridge(); }
    catch(const std::exception &e) { last_error=e.what(); return nullptr; }
}
void atec_matrix_close(void *p) { delete static_cast<Bridge*>(p); }
// Freeze our own controller child only while it owns none of the bridge locks.
// This lets a training agent pause physics and its low-level policy together.
int atec_matrix_pause(void *p, int pid) {
    try {
        auto &b=*static_cast<Bridge*>(p);
        bip::scoped_lock<bip::interprocess_mutex> a(*b.cmd.mutex);
        bip::scoped_lock<bip::interprocess_mutex> c(*b.state.mutex);
        bip::scoped_lock<bip::interprocess_mutex> d(*b.imu.mutex);
        if(kill(pid,SIGSTOP)) throw std::runtime_error("Cannot stop controller child");
        int status=0;
        for(int i=0;i<200;++i) {
            int result=waitpid(pid,&status,WUNTRACED|WNOHANG);
            if(result==pid && WIFSTOPPED(status)) return 0;
            if(result<0 || (result==pid && !WIFSTOPPED(status))) break;
            usleep(1000);
        }
        kill(pid,SIGCONT);
        throw std::runtime_error("Controller did not acknowledge stop");
    } catch(const std::exception &e) { last_error=e.what(); return -1; }
}
// Arrays use native encoder coordinates; Python owns the audited joint mapping.
int atec_matrix_write(void *p,const double *q,const double *v,const double *tor,
                      const double *acc,const double *gyro,const double *quat) {
    try {
        auto &b=*static_cast<Bridge*>(p); auto stamp=now_ms();
        {
            bip::scoped_lock<bip::interprocess_mutex> lock(*b.state.mutex);
            for(int i=0;i<16;++i) {
                auto &s=(*b.state.vec)[i]; s.timestamp_ms=stamp;
                s.pos=q[i]; s.vel=v[i]; s.tor=tor[i]; s.temp=30; s.voltage=48; s.enable=true;
            }
        }
        {
            bip::scoped_lock<bip::interprocess_mutex> lock(*b.imu.mutex);
            auto &s=(*b.imu.vec)[0]; s.timestamp_ms=stamp;
            std::memcpy(s.acc,acc,24); std::memcpy(s.gyro,gyro,24); std::memcpy(s.quat,quat,32);
        }
        b.state.condition->notify_all(); b.imu.condition->notify_all(); return 0;
    } catch(const std::exception &e) { last_error=e.what(); return -1; }
}
int atec_matrix_read(void *p,double *out,uint64_t *stamp) {
    try {
        auto &b=*static_cast<Bridge*>(p);
        bip::scoped_lock<bip::interprocess_mutex> lock(*b.cmd.mutex);
        for(int i=0;i<16;++i) {
            auto &c=(*b.cmd.vec)[i]; stamp[i]=c.timestamp_ms;
            out[5*i]=c.pos; out[5*i+1]=c.vel; out[5*i+2]=c.tor;
            out[5*i+3]=c.kp; out[5*i+4]=c.kd;
        }
        return 0;
    } catch(const std::exception &e) { last_error=e.what(); return -1; }
}
}
