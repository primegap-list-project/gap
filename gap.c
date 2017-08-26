// Search for large primegaps, written by Robert Gerbicz

//Fermat & Euler-Plumb PRP tests using Montgomery math
//written by Dana Jacobsen

#define version "1.05" // use (real) number!!!! do not put letters
                       // or other characters in the version name
					   
// version 1.05     Remove always false tests from prec_prime              Antonio Key
//					and next_prime, alternative would be to replace
//					with asserts as the progam failed elsewhere if
//					the tests were ever true.
//					Removed always true test in gap search
//					Removed bound test on P2 in gap search
//					Replaced result sort routine
//					On screen display of process state (sieve or gap search)
//					Let compiler decide if AVX2 is available to use
//					update default_unknowngap
//					Moved the #define version so that it was easier to check
//					against the version comments and keep them consistant.

// version 1.04     make a report using gap_solutions.txt file              Robert Gerbicz

// version 1.03     Save the step value for k                               Robert Gerbicz
//                  various cosmetic changes                                Antonio Key

// version 1.02     If you use the worktodo file at rerun, then             Robert Gerbicz
//                  use all parameters from that (so we don't see
//                  the command line for n1,n2,n,res1,res2,m1,m2)

// version 1.01     Several improvements (new sieve strategy etc.)          Robert Gerbicz

// version 0.05.g	Fixed quick sort, added optimized bubble sort.			Antonio Key
//					Replaced add30 & sub30 constant arrays with a single
//					wheel30 constant array - they became idential in 0.05.
//					Added ETA to status display line.
//					Removed some of the old commented-out code.
//					Modified the smart check on input data.

// version 0.05.f	next_prime, prec_prime tuning.							Antonio Key

// version 0.05.e	Hensel lifting.											Robert Gerbicz

// version 0.05.d	Fermat & Euler-Plumb PRP tests using Montgomery math.	Dana Jacobsen

// version 0.05		Bug fix.												Robert Gerbicz

// my long compilation line: gcc -flto -m64 -fopenmp -O2 -fomit-frame-pointer -mavx2 -mtune=skylake -march=skylake -o gap gap10.c -lm
// don't forget -fopenmp  [for OpenMP]
// use your own processor type, mine is skylake

/*[Antonio/]
my compile lines:
For pre-Haswell Core i processors (Nehalem to Ivybridge):
gcc -static -m64 -fopenmp -O2 -frename-registers -fomit-frame-pointer -flto -msse4.2 -mtune=nehalem -march=nehalem -o gap11 gap11.c -lm
For Haswell or later Core i processors:
gcc -static -m64 -fopenmp -O2 -frename-registers -fomit-frame-pointer -flto -mavx2 -mtune=haswell -march=haswell -o gap11_haswell gap11.c -lm
If your version of gcc supports later processors then you can substitute in -mtune and -march for the appropriate processor.
[\Antonio]
*/

// To help you for a given p1,p2=p1+gap where to rediscover the gap for given m1,m2,numcoprime,unknowngap values
// note: we don't check here that p1,p2=p1+gap are consecutive primes or not
// ( here we assume that we printed the gap in res!=res1, and used save_nextprimetest=1 )
// Pari-Gp scripts:
//getdelta(mid,m1,m2,v,numcoprime,unknowngap)={local(m,nc,d,w);w=vector(numcoprime);m=m1*m2;nc=0;d=0;while(nc<numcoprime&&d+m1<=unknowngap,if(gcd(v+d,m)==1,nc+=1;w[nc]=d);d+=1);if(mid,return(w[floor(nc/2)+2]),return(w[nc]))}
//findres(p1,gap,m1,m2,numcoprime,unknowngap)={local(p2,m,ret,v,d2,d1,d0);m=m1*m2;p2=p1+gap;ret=[];forstep(v=floor(p1/m1)*m1,p2,m1,d2=getdelta(0,m1,m2,v,numcoprime,unknowngap);d1=getdelta(1,m1,m2,v-m1,numcoprime,unknowngap);d0=getdelta(0,m1,m2,v-m1,numcoprime,unknowngap);if(p1<v&&p2>v+d2&&sum(t=0,d2,isprime(v+t))==0&&(p2-(v-m1+d1)>=unknowngap||sum(t=d1,d0,isprime(v-m1+t)==0)),ret=concat(ret,(v/m1)%m2)));return(ret)}

// example:
// findres(5295442011781310951,1448,1190,8151,27,1366)
// ans=[3684, 3685]
// so we can rediscover this gap for res=3684 and also for res=3685  (for m1=1190;m2=8151;numcoprime=27;unknowngap=1366)
// (it is possible to discover the same gap at different res values, it is common for larger gaps)

// 1 if using GCC compiler or other compiler that defines  __AVX2__ for appropriate processors
// 0 if compiler doesn't define __AVX2__ for appropriate processors
#if 1
	#if defined (__AVX2__)
		#define USE_AVX2 1
	#else
		#define USE_AVX2 0
	#endif
#else
	#define USE_AVX2 1 // use 0 if compiling for pre-Haswell processors
#endif
#define AVX2_BITS 256 // number of bits

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "omp.h" // for multithreading, need gcc >= 4.2

#if USE_AVX2
#include <immintrin.h>
#endif

typedef unsigned int           ui32;
typedef signed int             si32;
typedef long long int          si64;
typedef unsigned long long int ui64;

// these are only used to show as a hint, code will ask these as input/switch/file,
// so there is no need to modify these here.
#define default_sieve_bits_log2  25 // Total number of sieve array's bits(!!!), the most important constant
                            // say you have 3 MB of L3 cache, then sb=24 is a good choice since 2^24 bits=2 MB
                            // 2nd example:
                            // for 8 MB Intel's smart cache size use sb=25 (it is also the default here)
                            //    though it could seem a suboptimal choice, because 2^26 bits is exactly 8 MB
                            //    but test runs confirms that sb=25 is better than sb=26
                            // you can see this as "sb" in the code

#define default_bucket_size_log2 18 // in bytes, you can see this as "bs" in the code

#define default_unknowngap 1382 // currently the smallest unknown case is gap=1382
                                // see http://www.trnicely.net/gaps/gaplist.html
                                // use the -unknowngap to modify this (you can set even a lower number also)

#define default_numcoprime 27 // we will sieve numcoprime numbers per M1 in the k*mod+RES*M1+[0,M1) interval

// *******************
// For newer/older processors use your own better settings for the below constants
// modify these 6 constants here, we won't input this:
#define save_nextprimetest 1	// we use another (two!!) big arrays to save one next_prime() cost after the prec_prime()
								// in most of the cases. note that if you would use save_nextprimetest=0
								// to save memory (but get slower sieve!)

#define default_report_gap 1000	// we print and save gap=p2-p1 iff gap>=default_report_gap or gap>=unknowngap
                                // though it is NOT an exhustive search(!!!), only for >=unknowngap

#define count_LEN_intervals 32	// to lower some init costs in the sieve, 32 is quite a good choice
								// and changing to a much larger value would need a much longer
								// sieve array, so a much larger memory
								// note: this can be non-power of two also.

#define MAX_SIZE (1LL<<29)		// in bytes for offsets array, to save some space (0.5-1 GB) lower this
								// to say (1LL<<24), but that will give a slower sieve
								// this can be also non-power of two

#define MAX_NUM_SOLUTIONS 32  	// max. number of solutions per thread, we still can print/save results if
								// there would be more solutions

#define ALIGNEMENT 4096			// alignement (in bytes!)
								// it could be say (1<<bucket_size_log2), and at least 64.
								// or a higher power of two, but that gives no speedup
								// [it should be a power of two]
#define NUM_SIEVE_PRIMES 100    // for prec_prime, next_prime
                                // note that p=2 is excluded from the sieving primes! (so not counted)
                                // it should be divisible by 5, in the [5,165] interval
// *******************
void make_a_report(void);

double used_memory; // in gigabytes
double max_memory_gigabytes;// we'll set this, here 1 Gbyte=2^30 bytes=1073741824 bytes
                            // the code try to use no more memory than this.

int sieve_bits_log2; // Use the -sb switch to give it in the code
int bucket_size_log2;// Use the -bs switch
int gap_delta;
int MNS2 = 2 * MAX_NUM_SOLUTIONS;

#define inf64 0xffffffffffffffff // 2^64-1
#define inf63 0x7fffffffffffffff // 2^63-1
#define inf32 0xffffffff         // 2^32-1
#define inf31 0x7fffffff         // 2^31-1
#define DP30  1073741824.0       // 2.0^30
#define size_ui32 (sizeof(ui32)) // at least 4
#define size_ui64 (sizeof(ui64)) // it should be 8


#define get_lsb(a) (__builtin_ffsll(a)-1)
#define bitlen(a)  (64-__builtin_clzll(a))
#define msb(a)     (63-__builtin_clzll(a))


ui32 *inv_mod;
ui64 *isprime_table;
ui64 *bitmap;
ui64 step_k2;

static void print_err(void){
     printf("This is a newer code. Complete your previous range(s) with an earlier code.\n");
}

static void usage(void){
    printf("Usage: [program name] [options]\n");
    printf("\nOptions:\n");
    printf("  -n1 x          first number to check is x\n"); 
    printf("  -n2 y          last number to check is y\n"); 
    printf("  -n v           the test is at n=v\n");
    printf("  -res1 r1       first tested residue is r1\n");
    printf("  -res2 r2       first non-tested residue is r2, so we test the [r1,r2) interval.\n");
    printf("  -res r         the test is at res=r (for the given n)\n");
    printf("  -m1 x          use m1=x\n");
    printf("  -m2 y          use m2=y\n");
    printf("  -numcoprime v  sieving v numbers coprime to m in interval length=m1 (default=%d)\n",default_numcoprime);
    printf("  -sb u          sieve uses 2^u bits of memory (default=%d)\n",default_sieve_bits_log2);
    printf("  -bs v          one bucket size is 2^v bytes (default=%d)\n",default_bucket_size_log2);
    printf("  -mem m         the maximal memory usage is m GB (m can be any real number)\n");
    printf("  -t k           use k threads\n");
    printf("  -unknowngap d  smallest unknown gap is d (default=%d)\n",default_unknowngap);
    printf("  -makereport    make a report (we'll automatically do it after a finished run)\n");
}

int set_n1;
int set_n2;
int set_n;
int set_res1;
int set_res2;
int set_res;
int set_m1;
int set_m2;
int set_numcoprime;
int set_sb;
int set_bs;
int set_mem;
int set_t;
int set_stepk;
int set_makereport;
ui64 n0;

ui32 ppi;// number of primes, excluded prime divisors of mod and the small tablesieving primes
ui32 *primes;
ui32 *res_table;

int num_coprime,r_table[1024];
ui64 *offsets;
int threads;// we will input this

ui64 first_n,last_n;
ui32 primes_per_bucket;
ui32 num_sieve;
ui32 hash3,sh3,hash2,hash0;
ui32 sieve_length_bits_log2;
ui32 num_bucket;
ui32 LEN,LEN64;
ui32 cnt_smallprimes;
ui32 cnt_offsets;
ui64 PROD[64],SIZE[64];
ui32 PPI_131072;
ui32 PPI_LEN;
ui32 *TH;


int NUMCOPRIME;
ui32 M1,M2,RES,RES1,RES2;
ui64 mod;
int unknowngap;

static const ui64 Bits[64]={// Bits[i]=2^i
  0x0000000000000001,0x0000000000000002,0x0000000000000004,0x0000000000000008,
  0x0000000000000010,0x0000000000000020,0x0000000000000040,0x0000000000000080,
  0x0000000000000100,0x0000000000000200,0x0000000000000400,0x0000000000000800,
  0x0000000000001000,0x0000000000002000,0x0000000000004000,0x0000000000008000,
  0x0000000000010000,0x0000000000020000,0x0000000000040000,0x0000000000080000,
  0x0000000000100000,0x0000000000200000,0x0000000000400000,0x0000000000800000,
  0x0000000001000000,0x0000000002000000,0x0000000004000000,0x0000000008000000,
  0x0000000010000000,0x0000000020000000,0x0000000040000000,0x0000000080000000,
  0x0000000100000000,0x0000000200000000,0x0000000400000000,0x0000000800000000,
  0x0000001000000000,0x0000002000000000,0x0000004000000000,0x0000008000000000,
  0x0000010000000000,0x0000020000000000,0x0000040000000000,0x0000080000000000,
  0x0000100000000000,0x0000200000000000,0x0000400000000000,0x0000800000000000,
  0x0001000000000000,0x0002000000000000,0x0004000000000000,0x0008000000000000,
  0x0010000000000000,0x0020000000000000,0x0040000000000000,0x0080000000000000,
  0x0100000000000000,0x0200000000000000,0x0400000000000000,0x0800000000000000,
  0x1000000000000000,0x2000000000000000,0x4000000000000000,0x8000000000000000};

static const ui64 InvBits[64]={// InvBits[i]=2^64-1-2^i for masking
0xfffffffffffffffe,0xfffffffffffffffd,0xfffffffffffffffb,0xfffffffffffffff7,
0xffffffffffffffef,0xffffffffffffffdf,0xffffffffffffffbf,0xffffffffffffff7f,
0xfffffffffffffeff,0xfffffffffffffdff,0xfffffffffffffbff,0xfffffffffffff7ff,
0xffffffffffffefff,0xffffffffffffdfff,0xffffffffffffbfff,0xffffffffffff7fff,
0xfffffffffffeffff,0xfffffffffffdffff,0xfffffffffffbffff,0xfffffffffff7ffff,
0xffffffffffefffff,0xffffffffffdfffff,0xffffffffffbfffff,0xffffffffff7fffff,
0xfffffffffeffffff,0xfffffffffdffffff,0xfffffffffbffffff,0xfffffffff7ffffff,
0xffffffffefffffff,0xffffffffdfffffff,0xffffffffbfffffff,0xffffffff7fffffff,
0xfffffffeffffffff,0xfffffffdffffffff,0xfffffffbffffffff,0xfffffff7ffffffff,
0xffffffefffffffff,0xffffffdfffffffff,0xffffffbfffffffff,0xffffff7fffffffff,
0xfffffeffffffffff,0xfffffdffffffffff,0xfffffbffffffffff,0xfffff7ffffffffff,
0xffffefffffffffff,0xffffdfffffffffff,0xffffbfffffffffff,0xffff7fffffffffff,
0xfffeffffffffffff,0xfffdffffffffffff,0xfffbffffffffffff,0xfff7ffffffffffff,
0xffefffffffffffff,0xffdfffffffffffff,0xffbfffffffffffff,0xff7fffffffffffff,
0xfeffffffffffffff,0xfdffffffffffffff,0xfbffffffffffffff,0xf7ffffffffffffff,
0xefffffffffffffff,0xdfffffffffffffff,0xbfffffffffffffff,0x7fffffffffffffff};

static const int wheel30[8]={2,6,4,2,4,2,4,6};

// (i*inv64[i])%64==1 for odd i
#if USE_AVX2
static const int inv_table[256]={
0,1,0,171,0,205,0,183,0,57,0,163,0,197,0,239,0,241,0,27,0,61,0,167,0,41,0,19,0,53,
0,223,0,225,0,139,0,173,0,151,0,25,0,131,0,165,0,207,0,209,0,251,0,29,0,135,0,9,0,243,
0,21,0,191,0,193,0,107,0,141,0,119,0,249,0,99,0,133,0,175,0,177,0,219,0,253,0,103,0,233,
0,211,0,245,0,159,0,161,0,75,0,109,0,87,0,217,0,67,0,101,0,143,0,145,0,187,0,221,0,71,
0,201,0,179,0,213,0,127,0,129,0,43,0,77,0,55,0,185,0,35,0,69,0,111,0,113,0,155,0,189,
0,39,0,169,0,147,0,181,0,95,0,97,0,11,0,45,0,23,0,153,0,3,0,37,0,79,0,81,0,123,
0,157,0,7,0,137,0,115,0,149,0,63,0,65,0,235,0,13,0,247,0,121,0,227,0,5,0,47,0,49,
0,91,0,125,0,231,0,105,0,83,0,117,0,31,0,33,0,203,0,237,0,215,0,89,0,195,0,229,0,15,
0,17,0,59,0,93,0,199,0,73,0,51,0,85,0,255};
#else
static const int inv_table[64]={0,1,0,43,0,13,0,55,0,57,0,35,0,5,0,47,0,49,0,27,0,61,0,39,0,41,0,19,0,53,
0,31,0,33,0,11,0,45,0,23,0,25,0,3,0,37,0,15,0,17,0,59,0,29,0,7,0,9,0,51,0,21,0,63};
#endif

typedef struct{
ui32 offset;
ui32 pr;  // prime number
}bucket;

typedef struct {
ui32 gap;
ui64 p1;
}GAP;

void print_time(void){
    time_t timer;
    char w[128];
    struct tm* tm_info;
    time(&timer);
    tm_info=localtime(&timer);
    strftime(w,128,"%Y-%m-%d %H:%M:%S",tm_info);
    printf("%s",w);
}

void print_error_msg(void){printf("Not enough memory!!!\n");exit(1);}
ui64 imax64(ui64 x,ui64 y){if(x>y)return x;return y;}
ui64 imin64(ui64 x,ui64 y){if(x<y)return x;return y;}
ui64 gcd64(ui64 x,ui64 y){if(y==0)return x;return gcd64(y,x%y);}
ui64 lcm64(ui64 x,ui64 y){return (x/gcd64(x,y))*y;}// possibly overflow
ui32 is_power_two(ui64 n){return (n>0&&(n&(n-1))==0);}// return 1 if n is a power of two
ui64 mround_gen(ui64 n,ui64 m){return m*((n+m-1)/m);}
ui64 mround_512(ui64 n){return mround_gen(n,512);}// return the smallest multiple of 512, not smaller than n
ui64 precpower2(ui64 n){return Bits[bitlen(n)-1];}
ui64 nextpower2(ui64 n){if(n<2)return 1;return Bits[bitlen(n-1)];}

ui32 primes2[172]={2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,
59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,
137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,
227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,
313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,
419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,
509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,
617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,
727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,
829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,
947,953,967,971,977,983,991,997,1009,1013,1019,1021};// primes up to 1024


ui64 conv64(char w[]){
    int i,len=strlen(w);
    for(i=0;i<len;i++)
        if(w[i]=='.'||w[i]=='e'||w[i]=='E'){double d;sscanf(w,"%lf",&d);return ((ui64)d);}
    ui64 v;
    sscanf(w,"%llu",&v);
    return v;
}

ui32 lin_solve(ui64 res,ui32 p){
// Solve x*mod+res==0 modulo p
// assume that gcd(p,mod)=1 & M1*M2=mod
// x==-res/mod modulo p
// (note: res>=p is possible in the input)
   ui64 k=(ui64)p*inv_mod[p%mod];
   k=(k-1)/mod;
   res=((ui64)(res%p)*k)%p;// we need res%p to avoid overflow in 64 bits
   return ((ui32)res);
}

static double cpu_time(void){// known small code
  struct rusage r;
  double t;
  getrusage(RUSAGE_SELF,&r);
  t=(double)(r.ru_utime.tv_sec+r.ru_stime.tv_sec);
  t+=1.0e-6*(double)(r.ru_utime.tv_usec+r.ru_stime.tv_usec);
  return t;
}

ui32 isqrt64(ui64 n){// integer square root of n, speed is not important, we call it very few times

     ui32 ret,v,p2;
     for(ret=0,p2=inf31;p2;p2>>=1){
         v=ret+p2;
         if(((ui64)v*v)<=n)ret=v;
     }
     return ret;
}

ui64 mulmod(ui64 a,ui64 b,ui64 n) {// (a*b)%n
// from mersenneforum.org, with small modification: use mulq,divq instead of imulq,idivq
    ui64 d, dummy; /* d will get a*b mod c */
    asm ("mulq %3\n\t" /* mul a*b -> rdx:rax */
         "divq %4\n\t" /* (a*b)/c -> quot in rax remainder in rdx */
         :"=a"(dummy), "=&d"(d) /* output */
         :"a"(a), "rm"(b), "rm"(n) /* input */
         :"cc" /* imulq and idivq can set conditions */
        );
    return d;
}

#if 0
  /* Gerbicz */
ui64 fermatpowmod2_63(ui64 n){// n>0, return (2^(n-1)) mod n, assume that n<2^63

    if(n<=64)return (Bits[n-1]%n);

    si32 i;
    ui64 n1=n-1;
    ui32 len=bitlen(n1);
    ui64 ret=Bits[n1>>(len-6)]%n,n2=n1>>1;

    for(i=len-7;i>=0;i--){
        ret=mulmod(ret,ret,n);
        if((n1>>i)&1){//do ret=(2*ret)%n;
            if(ret>n2)ret=(ret<<1)-n;
            else      ret<<=1;
        }
    }
    return ret;
}

ui64 fermatpowmod2(ui64 n){// return (2^(n-1)) mod n

    if(n<inf63)return fermatpowmod2_63(n);

    si32 i;
    ui64 n1=n-1;
    ui32 len=bitlen(n1);
    ui64 ret=Bits[n1>>(len-6)]%n,n2=n1>>1;

    for(i=len-7;i>=0;i--){
        ret=mulmod(ret,ret,n);
        if((n1>>i)&1){
            if(ret>n2)ret=ret-(n-ret);// to avoid overflow in 64 bits
            else      ret<<=1;
        }
    }
    return ret;
}

ui32 fermatprp2(ui64 n){return (fermatpowmod2(n)==1);}
#else
  /* Jacobsen / Izykowski */
  static inline ui64 addmod(ui64 a, ui64 b, ui64 n) {
    ui64 r = a+b;
    ui64 t = a-n;
    asm ("add %2, %1\n\t"    /* t := t + b */
         "cmovc %1, %0\n\t"  /* if (carry) r := t */
         :"+r" (r), "+&r" (t)
         :"rm" (b)
         :"cc"
        );
    return r;
  }
#define MPU_UNLIKELY(x) __builtin_expect(!!(x), 0)
static inline ui64 mont_prod64(ui64 a, ui64 b, ui64 n, ui64 npi)
{
  ui64 t_hi, t_lo, m, mn_hi, mn_lo, u;
  /* t_hi * 2^64 + t_lo = a*b */
  asm("mulq %3" : "=a"(t_lo), "=d"(t_hi) : "a"(a), "rm"(b));
  if (MPU_UNLIKELY(t_lo == 0)) return t_hi;  /* Allows carry=1 below */
  m = t_lo * npi;
  /* mn_hi * 2^64 + mn_lo = m*n */
  asm("mulq %3" : "=a"(mn_lo), "=d"(mn_hi) : "a"(m), "rm"(n));
  u = t_hi + mn_hi + 1;
  return (u < t_hi || u >= n)  ?  u-n  :  u;
}
#define mont_square64(a, n, npi)  mont_prod64(a, a, n, npi)
static inline ui64 mont_powmod64(ui64 a, ui64 k, ui64 one, ui64 n, ui64 npi)
{
  ui64 t = one;
  while (k) {
    if (k & 1) t = mont_prod64(t, a, n, npi);
    k >>= 1;
    if (k)     a = mont_square64(a, n, npi);
  }
  return t;
}

static inline ui64 mont_powmod64_base2(ui64 a, ui64 k, ui64 n, ui64 npi){
// do not use it for k=0, also, this routine assumes that we compute the montgomery powmod of two
  ui64 t = a;
  si32 i;
  for(i=bitlen(k)-2;i>=0;i--){
      t=mont_prod64(t,t,n,npi);
      if((k>>i)&1)t=addmod(t,t,n);
  }
  return t;
}

static inline ui64 modular_inverse64(const ui64 a)
{
	static const char mask[128] = {255,85,51,73,199,93,59,17,15,229,195,89,215,237,203,33,
	31,117,83,105,231,125,91,49,47,5,227,121,247,13,235,65,63,149,115,137,7,157,123,81,79,
	37,3,153,23,45,11,97,95,181,147,169,39,189,155,113,111,69,35,185,55,77,43,129,127,213,
	179,201,71,221,187,145,143,101,67,217,87,109,75,161,159,245,211,233,103,253,219,177,175,
	133,99,249,119,141,107,193,191,21,243,9,135,29,251,209,207,165,131,25,151,173,139,225,
	223,53,19,41,167,61,27,241,239,197,163,57,183,205,171,1};

	#if 1
    // Gerbicz - Hensel lifting
    ui64 ret = mask[(a >> 1) & 127];
    ret *= 2 + a * ret;
    ret *= 2 + a * ret;
    ret *= 2 + a * ret;
    return ret;
	#else
	// Jacobsen
	/* Basic algorithm:
	*    for (i = 0; i < 64; i++) {
	*      if (S & 1)  {  J |= (1ULL << i);  S += a;  }
	*      S >>= 1;
	*    }
	* What follows is 8 bits at a time, unrolled by hand. */
	ui64 J, S = 1;
	ui32 T;
	int idx;
	const char amask = mask[(a >> 1) & 127];
	idx = (amask*(S&255)) & 255;  J = idx;              S = (S+a*idx) >> 8;
	idx = (amask*(S&255)) & 255;  J |= (ui64)idx << 8;  S = (S+a*idx) >> 8;
	idx = (amask*(S&255)) & 255;  J |= (ui64)idx <<16;  S = (S+a*idx) >> 8;
	idx = (amask*(S&255)) & 255;  J |= (ui64)idx <<24;  T = (S+a*idx) >> 8;
	idx = (amask*(T&255)) & 255;  J |= (ui64)idx <<32;  T = (T+a*idx) >> 8;
	idx = (amask*(T&255)) & 255;  J |= (ui64)idx <<40;  T = (T+a*idx) >> 8;
	idx = (amask*(T&255)) & 255;  J |= (ui64)idx <<48;  T = (T+a*idx) >> 8;
	idx = (amask*(T&255)) & 255;  J |= (ui64)idx <<56;
	return J;
	#endif
}

static inline ui64 compute_modn64(const ui64 n)
{

  if (n <= (1ULL << 63)) {
    ui64 res = ((1ULL << 63) % n) << 1;
    return res < n ? res : res-n;
  } else
    return -n;
}
#define mont_inverse(n)           modular_inverse64(n)
#define mont_get1(n)              compute_modn64(n)
#define mont_get2(n)              addmod(mont1,mont1,n)
#define mont_powmod(a,k,n)        mont_powmod64(a,k,mont1,n,npi)
#define mont_powmod2(a,k,n)       mont_powmod64_base2(a,k,n,npi)          
ui32 fermatprp2(ui64 n){
  //return (fermatpowmod2(n)==1);
  /* n must be odd */
  #if 0  /* Fermat */
    const ui64 npi = mont_inverse(n),  mont1 = mont_get1(n);
    const ui64 mont2 = mont_get2(n);
    return mont_powmod(mont2, n-1, n) == mont1;
  #else /* Euler-Plumb */
    const ui64 npi = mont_inverse(n),  mont1 = mont_get1(n);
    const ui64 mont2 = mont_get2(n);
    ui32 nmod8 = n & 0x7;
    ui64 ap = mont_powmod2(mont2, (n-1) >> (1 + (nmod8 == 1)), n);
	if (ap ==   mont1)  return (nmod8 == 1 || nmod8 == 7);
    if (ap == n-mont1)  return (nmod8 == 1 || nmod8 == 3 || nmod8 == 5);
    return 0;
  #endif
}
#endif

#define trial_div(n)	(n%7&&n%11&&n%13&&n%17&&n%19&&n%23&&n%29&&n%31&&n%37&&n%41&&n%43&&n%47&&n%53&&n%59&&\
						 n%61&&n%67&&n%71&&n%73&&n%79&&n%83&&n%89&&n%97&&n%101&&n%103&&n%107&&n%109&&n%113&&\
						 n%127&&n%131&&n%137&&n%139&&n%149&&n%151&&n%157&&n%163&&n%167&&n%173&&n%179&&n%181&&\
						 n%191&&n%193&&n%197&&n%199&&n%211&&n%223&&n%227&&n%229&&n%233&&n%239&&n%241&&n%251&&\
						 n%257&&n%263&&n%269&&n%271&&n%277&&n%281&&n%283&&n%293&&n%307&&n%311&&n%313&&n%317&&fermatprp2(n))


// currently we use method 2 for next/prec prime routines; (currently) this is the faster method
#define next_prime(n)    next_prime2(n)
#define prec_prime(n)    prec_prime2(n)

ui64 next_prime1(ui64 n){
    ui32 i = 0;
    ui64 n2=n-1-(n%30);
	while (n2<n) {n2+=wheel30[i&7]; i++;}
	do {if(trial_div(n2))return n2; n2+=wheel30[i&7];} while (i++);
}

ui64 prec_prime1(ui64 n){
    ui32 i = 0;
    ui64 n2=n+31-(n%30);
	while (n2>n) {n2-=wheel30[i&7]; i++;}
	do {if(trial_div(n2))return n2; n2-=wheel30[i&7];} while (i++);
}

ui64 next_prime2(ui64 n){// note: do not call it for n<1024
    ui32 shift;
    ui64 v,n2=n+((n+1)&1);
    for(;;n2+=128){
    ui64 *b=bitmap;
    
#if NUM_SIEVE_PRIMES>=5
v=b[n2%3];b+=3;v&=b[n2%5];b+=5;v&=b[n2%7];b+=7;v&=b[n2%11];b+=11;v&=b[n2%13];b+=13;
#endif
#if NUM_SIEVE_PRIMES>=10
v&=b[n2%17];b+=17;v&=b[n2%19];b+=19;v&=b[n2%23];b+=23;v&=b[n2%29];b+=29;v&=b[n2%31];b+=31;
#endif
#if NUM_SIEVE_PRIMES>=15
v&=b[n2%37];b+=37;v&=b[n2%41];b+=41;v&=b[n2%43];b+=43;v&=b[n2%47];b+=47;v&=b[n2%53];b+=53;
#endif
#if NUM_SIEVE_PRIMES>=20
v&=b[n2%59];b+=59;v&=b[n2%61];b+=61;v&=b[n2%67];b+=67;v&=b[n2%71];b+=71;v&=b[n2%73];b+=73;
#endif
#if NUM_SIEVE_PRIMES>=25
v&=b[n2%79];b+=79;v&=b[n2%83];b+=83;v&=b[n2%89];b+=89;v&=b[n2%97];b+=97;v&=b[n2%101];b+=101;
#endif
#if NUM_SIEVE_PRIMES>=30
v&=b[n2%103];b+=103;v&=b[n2%107];b+=107;v&=b[n2%109];b+=109;v&=b[n2%113];b+=113;v&=b[n2%127];b+=127;
#endif
#if NUM_SIEVE_PRIMES>=35
v&=b[n2%131];b+=131;v&=b[n2%137];b+=137;v&=b[n2%139];b+=139;v&=b[n2%149];b+=149;v&=b[n2%151];b+=151;
#endif
#if NUM_SIEVE_PRIMES>=40
v&=b[n2%157];b+=157;v&=b[n2%163];b+=163;v&=b[n2%167];b+=167;v&=b[n2%173];b+=173;v&=b[n2%179];b+=179;
#endif
#if NUM_SIEVE_PRIMES>=45
v&=b[n2%181];b+=181;v&=b[n2%191];b+=191;v&=b[n2%193];b+=193;v&=b[n2%197];b+=197;v&=b[n2%199];b+=199;
#endif
#if NUM_SIEVE_PRIMES>=50
v&=b[n2%211];b+=211;v&=b[n2%223];b+=223;v&=b[n2%227];b+=227;v&=b[n2%229];b+=229;v&=b[n2%233];b+=233;
#endif
#if NUM_SIEVE_PRIMES>=55
v&=b[n2%239];b+=239;v&=b[n2%241];b+=241;v&=b[n2%251];b+=251;v&=b[n2%257];b+=257;v&=b[n2%263];b+=263;
#endif
#if NUM_SIEVE_PRIMES>=60
v&=b[n2%269];b+=269;v&=b[n2%271];b+=271;v&=b[n2%277];b+=277;v&=b[n2%281];b+=281;v&=b[n2%283];b+=283;
#endif
#if NUM_SIEVE_PRIMES>=65
v&=b[n2%293];b+=293;v&=b[n2%307];b+=307;v&=b[n2%311];b+=311;v&=b[n2%313];b+=313;v&=b[n2%317];b+=317;
#endif
#if NUM_SIEVE_PRIMES>=70
v&=b[n2%331];b+=331;v&=b[n2%337];b+=337;v&=b[n2%347];b+=347;v&=b[n2%349];b+=349;v&=b[n2%353];b+=353;
#endif
#if NUM_SIEVE_PRIMES>=75
v&=b[n2%359];b+=359;v&=b[n2%367];b+=367;v&=b[n2%373];b+=373;v&=b[n2%379];b+=379;v&=b[n2%383];b+=383;
#endif
#if NUM_SIEVE_PRIMES>=80
v&=b[n2%389];b+=389;v&=b[n2%397];b+=397;v&=b[n2%401];b+=401;v&=b[n2%409];b+=409;v&=b[n2%419];b+=419;
#endif
#if NUM_SIEVE_PRIMES>=85
v&=b[n2%421];b+=421;v&=b[n2%431];b+=431;v&=b[n2%433];b+=433;v&=b[n2%439];b+=439;v&=b[n2%443];b+=443;
#endif
#if NUM_SIEVE_PRIMES>=90
v&=b[n2%449];b+=449;v&=b[n2%457];b+=457;v&=b[n2%461];b+=461;v&=b[n2%463];b+=463;v&=b[n2%467];b+=467;
#endif
#if NUM_SIEVE_PRIMES>=95
v&=b[n2%479];b+=479;v&=b[n2%487];b+=487;v&=b[n2%491];b+=491;v&=b[n2%499];b+=499;v&=b[n2%503];b+=503;
#endif
#if NUM_SIEVE_PRIMES>=100
v&=b[n2%509];b+=509;v&=b[n2%521];b+=521;v&=b[n2%523];b+=523;v&=b[n2%541];b+=541;v&=b[n2%547];b+=547;
#endif
#if NUM_SIEVE_PRIMES>=105
v&=b[n2%557];b+=557;v&=b[n2%563];b+=563;v&=b[n2%569];b+=569;v&=b[n2%571];b+=571;v&=b[n2%577];b+=577;
#endif
#if NUM_SIEVE_PRIMES>=110
v&=b[n2%587];b+=587;v&=b[n2%593];b+=593;v&=b[n2%599];b+=599;v&=b[n2%601];b+=601;v&=b[n2%607];b+=607;
#endif
#if NUM_SIEVE_PRIMES>=115
v&=b[n2%613];b+=613;v&=b[n2%617];b+=617;v&=b[n2%619];b+=619;v&=b[n2%631];b+=631;v&=b[n2%641];b+=641;
#endif
#if NUM_SIEVE_PRIMES>=120
v&=b[n2%643];b+=643;v&=b[n2%647];b+=647;v&=b[n2%653];b+=653;v&=b[n2%659];b+=659;v&=b[n2%661];b+=661;
#endif
#if NUM_SIEVE_PRIMES>=125
v&=b[n2%673];b+=673;v&=b[n2%677];b+=677;v&=b[n2%683];b+=683;v&=b[n2%691];b+=691;v&=b[n2%701];b+=701;
#endif
#if NUM_SIEVE_PRIMES>=130
v&=b[n2%709];b+=709;v&=b[n2%719];b+=719;v&=b[n2%727];b+=727;v&=b[n2%733];b+=733;v&=b[n2%739];b+=739;
#endif
#if NUM_SIEVE_PRIMES>=135
v&=b[n2%743];b+=743;v&=b[n2%751];b+=751;v&=b[n2%757];b+=757;v&=b[n2%761];b+=761;v&=b[n2%769];b+=769;
#endif
#if NUM_SIEVE_PRIMES>=140
v&=b[n2%773];b+=773;v&=b[n2%787];b+=787;v&=b[n2%797];b+=797;v&=b[n2%809];b+=809;v&=b[n2%811];b+=811;
#endif
#if NUM_SIEVE_PRIMES>=145
v&=b[n2%821];b+=821;v&=b[n2%823];b+=823;v&=b[n2%827];b+=827;v&=b[n2%829];b+=829;v&=b[n2%839];b+=839;
#endif
#if NUM_SIEVE_PRIMES>=150
v&=b[n2%853];b+=853;v&=b[n2%857];b+=857;v&=b[n2%859];b+=859;v&=b[n2%863];b+=863;v&=b[n2%877];b+=877;
#endif
#if NUM_SIEVE_PRIMES>=155
v&=b[n2%881];b+=881;v&=b[n2%883];b+=883;v&=b[n2%887];b+=887;v&=b[n2%907];b+=907;v&=b[n2%911];b+=911;
#endif
#if NUM_SIEVE_PRIMES>=160
v&=b[n2%919];b+=919;v&=b[n2%929];b+=929;v&=b[n2%937];b+=937;v&=b[n2%941];b+=941;v&=b[n2%947];b+=947;
#endif
#if NUM_SIEVE_PRIMES>=165
v&=b[n2%953];b+=953;v&=b[n2%967];b+=967;v&=b[n2%971];b+=971;v&=b[n2%977];b+=977;v&=b[n2%983];b+=983;
#endif
    
    for(;v;){
        shift=__builtin_ctzll(v);
		v-=Bits[shift];
        if(fermatprp2(n2+2*shift))return(n2+2*shift);
    }}
}

ui64 prec_prime2(ui64 n){// note: do not call it for n<1024
    ui32 shift;
    ui64 v,n2,m2=n-((n+1)&1);
    for(;;m2-=128){
    ui64 *b=bitmap;
    n2=m2-126;
    
#if NUM_SIEVE_PRIMES>=5
v=b[n2%3];b+=3;v&=b[n2%5];b+=5;v&=b[n2%7];b+=7;v&=b[n2%11];b+=11;v&=b[n2%13];b+=13;
#endif
#if NUM_SIEVE_PRIMES>=10
v&=b[n2%17];b+=17;v&=b[n2%19];b+=19;v&=b[n2%23];b+=23;v&=b[n2%29];b+=29;v&=b[n2%31];b+=31;
#endif
#if NUM_SIEVE_PRIMES>=15
v&=b[n2%37];b+=37;v&=b[n2%41];b+=41;v&=b[n2%43];b+=43;v&=b[n2%47];b+=47;v&=b[n2%53];b+=53;
#endif
#if NUM_SIEVE_PRIMES>=20
v&=b[n2%59];b+=59;v&=b[n2%61];b+=61;v&=b[n2%67];b+=67;v&=b[n2%71];b+=71;v&=b[n2%73];b+=73;
#endif
#if NUM_SIEVE_PRIMES>=25
v&=b[n2%79];b+=79;v&=b[n2%83];b+=83;v&=b[n2%89];b+=89;v&=b[n2%97];b+=97;v&=b[n2%101];b+=101;
#endif
#if NUM_SIEVE_PRIMES>=30
v&=b[n2%103];b+=103;v&=b[n2%107];b+=107;v&=b[n2%109];b+=109;v&=b[n2%113];b+=113;v&=b[n2%127];b+=127;
#endif
#if NUM_SIEVE_PRIMES>=35
v&=b[n2%131];b+=131;v&=b[n2%137];b+=137;v&=b[n2%139];b+=139;v&=b[n2%149];b+=149;v&=b[n2%151];b+=151;
#endif
#if NUM_SIEVE_PRIMES>=40
v&=b[n2%157];b+=157;v&=b[n2%163];b+=163;v&=b[n2%167];b+=167;v&=b[n2%173];b+=173;v&=b[n2%179];b+=179;
#endif
#if NUM_SIEVE_PRIMES>=45
v&=b[n2%181];b+=181;v&=b[n2%191];b+=191;v&=b[n2%193];b+=193;v&=b[n2%197];b+=197;v&=b[n2%199];b+=199;
#endif
#if NUM_SIEVE_PRIMES>=50
v&=b[n2%211];b+=211;v&=b[n2%223];b+=223;v&=b[n2%227];b+=227;v&=b[n2%229];b+=229;v&=b[n2%233];b+=233;
#endif
#if NUM_SIEVE_PRIMES>=55
v&=b[n2%239];b+=239;v&=b[n2%241];b+=241;v&=b[n2%251];b+=251;v&=b[n2%257];b+=257;v&=b[n2%263];b+=263;
#endif
#if NUM_SIEVE_PRIMES>=60
v&=b[n2%269];b+=269;v&=b[n2%271];b+=271;v&=b[n2%277];b+=277;v&=b[n2%281];b+=281;v&=b[n2%283];b+=283;
#endif
#if NUM_SIEVE_PRIMES>=65
v&=b[n2%293];b+=293;v&=b[n2%307];b+=307;v&=b[n2%311];b+=311;v&=b[n2%313];b+=313;v&=b[n2%317];b+=317;
#endif
#if NUM_SIEVE_PRIMES>=70
v&=b[n2%331];b+=331;v&=b[n2%337];b+=337;v&=b[n2%347];b+=347;v&=b[n2%349];b+=349;v&=b[n2%353];b+=353;
#endif
#if NUM_SIEVE_PRIMES>=75
v&=b[n2%359];b+=359;v&=b[n2%367];b+=367;v&=b[n2%373];b+=373;v&=b[n2%379];b+=379;v&=b[n2%383];b+=383;
#endif
#if NUM_SIEVE_PRIMES>=80
v&=b[n2%389];b+=389;v&=b[n2%397];b+=397;v&=b[n2%401];b+=401;v&=b[n2%409];b+=409;v&=b[n2%419];b+=419;
#endif
#if NUM_SIEVE_PRIMES>=85
v&=b[n2%421];b+=421;v&=b[n2%431];b+=431;v&=b[n2%433];b+=433;v&=b[n2%439];b+=439;v&=b[n2%443];b+=443;
#endif
#if NUM_SIEVE_PRIMES>=90
v&=b[n2%449];b+=449;v&=b[n2%457];b+=457;v&=b[n2%461];b+=461;v&=b[n2%463];b+=463;v&=b[n2%467];b+=467;
#endif
#if NUM_SIEVE_PRIMES>=95
v&=b[n2%479];b+=479;v&=b[n2%487];b+=487;v&=b[n2%491];b+=491;v&=b[n2%499];b+=499;v&=b[n2%503];b+=503;
#endif
#if NUM_SIEVE_PRIMES>=100
v&=b[n2%509];b+=509;v&=b[n2%521];b+=521;v&=b[n2%523];b+=523;v&=b[n2%541];b+=541;v&=b[n2%547];b+=547;
#endif
#if NUM_SIEVE_PRIMES>=105
v&=b[n2%557];b+=557;v&=b[n2%563];b+=563;v&=b[n2%569];b+=569;v&=b[n2%571];b+=571;v&=b[n2%577];b+=577;
#endif
#if NUM_SIEVE_PRIMES>=110
v&=b[n2%587];b+=587;v&=b[n2%593];b+=593;v&=b[n2%599];b+=599;v&=b[n2%601];b+=601;v&=b[n2%607];b+=607;
#endif
#if NUM_SIEVE_PRIMES>=115
v&=b[n2%613];b+=613;v&=b[n2%617];b+=617;v&=b[n2%619];b+=619;v&=b[n2%631];b+=631;v&=b[n2%641];b+=641;
#endif
#if NUM_SIEVE_PRIMES>=120
v&=b[n2%643];b+=643;v&=b[n2%647];b+=647;v&=b[n2%653];b+=653;v&=b[n2%659];b+=659;v&=b[n2%661];b+=661;
#endif
#if NUM_SIEVE_PRIMES>=125
v&=b[n2%673];b+=673;v&=b[n2%677];b+=677;v&=b[n2%683];b+=683;v&=b[n2%691];b+=691;v&=b[n2%701];b+=701;
#endif
#if NUM_SIEVE_PRIMES>=130
v&=b[n2%709];b+=709;v&=b[n2%719];b+=719;v&=b[n2%727];b+=727;v&=b[n2%733];b+=733;v&=b[n2%739];b+=739;
#endif
#if NUM_SIEVE_PRIMES>=135
v&=b[n2%743];b+=743;v&=b[n2%751];b+=751;v&=b[n2%757];b+=757;v&=b[n2%761];b+=761;v&=b[n2%769];b+=769;
#endif
#if NUM_SIEVE_PRIMES>=140
v&=b[n2%773];b+=773;v&=b[n2%787];b+=787;v&=b[n2%797];b+=797;v&=b[n2%809];b+=809;v&=b[n2%811];b+=811;
#endif
#if NUM_SIEVE_PRIMES>=145
v&=b[n2%821];b+=821;v&=b[n2%823];b+=823;v&=b[n2%827];b+=827;v&=b[n2%829];b+=829;v&=b[n2%839];b+=839;
#endif
#if NUM_SIEVE_PRIMES>=150
v&=b[n2%853];b+=853;v&=b[n2%857];b+=857;v&=b[n2%859];b+=859;v&=b[n2%863];b+=863;v&=b[n2%877];b+=877;
#endif
#if NUM_SIEVE_PRIMES>=155
v&=b[n2%881];b+=881;v&=b[n2%883];b+=883;v&=b[n2%887];b+=887;v&=b[n2%907];b+=907;v&=b[n2%911];b+=911;
#endif
#if NUM_SIEVE_PRIMES>=160
v&=b[n2%919];b+=919;v&=b[n2%929];b+=929;v&=b[n2%937];b+=937;v&=b[n2%941];b+=941;v&=b[n2%947];b+=947;
#endif
#if NUM_SIEVE_PRIMES>=165
v&=b[n2%953];b+=953;v&=b[n2%967];b+=967;v&=b[n2%971];b+=971;v&=b[n2%977];b+=977;v&=b[n2%983];b+=983;
#endif

    for(;v;){
        shift=msb(v);
        v-=Bits[shift];
        if(fermatprp2(n2+2*shift))return (n2+2*shift);
    }}
}

si64 single_modinv(si64 a,si64 modulus)
// return by a^(-1) mod modulus, from mersenneforum.org, Robert D. Silverman's code
{/* start of single_modinv */

 a%=modulus;
 if(modulus<0)  modulus=-modulus;
 if(a<0)  a+=modulus;

 si64 ps1, ps2, dividend, divisor, rem, q, t;
 ui32 parity;

 q = 1;
 rem = a;
 dividend = modulus;
 divisor = a;
 ps1 = 1;
 ps2 = 0;
 parity = 0;

 while (divisor > 1)
 {
 rem = dividend - divisor;
 t = rem - divisor;
 if (t >= 0) {
   q += ps1;
   rem = t;
   t -= divisor;
   if (t >= 0) {
     q += ps1;
     rem = t;
     t -= divisor;
     if (t >= 0) {
       q += ps1;
       rem = t;
       t -= divisor;
       if (t >= 0) {
         q += ps1;
         rem = t;
         t -= divisor;
         if (t >= 0) {
           q += ps1;
           rem = t;
           t -= divisor;
           if (t >= 0) {
             q += ps1;
             rem = t;
             t -= divisor;
             if (t >= 0) {
               q += ps1;
               rem = t;
               t -= divisor;
               if (t >= 0) {
                 q += ps1;
                 rem = t;
                 if (rem >= divisor) {
                   q = dividend/divisor;
                   rem = dividend - q * divisor;
                   q *= ps1;
                 }}}}}}}}}
 q += ps2;
 parity = ~parity;
 dividend = divisor;
 divisor = rem;
 ps2 = ps1;
 ps1 = q;
 }

 if(parity==0)
 return (ps1);
 else
 return (modulus - ps1);
}

typedef struct {
ui32 gap,res;
ui64 p1;
}GAP2;

int comp2(const void *a,const void *b){
  GAP2* g1=(GAP2*)a;
  GAP2* g2=(GAP2*)b;
  if((g1->res)>(g2->res))   return 1;
  if((g1->res)<(g2->res))   return (-1);
  if((g1->gap)>(g2->gap))   return 1;
  if((g1->gap)<(g2->gap))   return (-1);
  return 0;
}

void make_a_report(void){
    
    FILE* fin;
    fin=fopen("gap_solutions.txt","r");
    
    int gap,i,res,cnt=0,size=1,mingap,*bestgap;
    ui32 r0,r1;
    ui64 p1,p2,m=(ui64)M1*M2;
    GAP2 *g;
    g=(GAP2*)malloc(size*sizeof(GAP2));
    bestgap=(int*)malloc(M2*sizeof(int));
    
    mingap=imin64(default_report_gap,unknowngap);
    for(i=0;i<M2;i++)bestgap[i]=0;
    while(fscanf(fin,"%u %llu",&gap,&p1)!=EOF){
        p2=p1+gap;
        r0=(1+(p1%m)/M1)%M2;
        r1=(r0+1)%M2;
        if(p2>=first_n&&p2<=last_n&&gap>=mingap&&((r0>=RES1&&r0<RES2)||(r1>=RES1&&r1<RES2))){
           cnt++;
           if(cnt>size){size*=2;g=(GAP2*)realloc(g,size*sizeof(GAP2));}
           if(r0>=RES1&&r0<RES2)res=r0;
           else                 res=r1;
           g[cnt-1].gap=gap;
           g[cnt-1].p1=p1;
           g[cnt-1].res=res;
           bestgap[res]=imax64(bestgap[res],gap);
        }
    }
    fclose(fin);
    qsort(g,cnt,sizeof(*g),comp2);
    
    int ns=0;
    for(i=0;i<cnt;i++)if(i==0||g[i].p1!=g[i-1].p1)ns++;
    
    FILE* fout;
    fout=fopen("gap_report.txt","w");
    fprintf(fout,"Found %d different gaps; mingap=%d; interval=[%llu,%llu]; m1=%u;m2=%u; res=[%u,%u).\n",
            ns,mingap,first_n,last_n,M1,M2,RES1,RES2);
    
    int best=0;
    for(i=RES1;i<RES2;i++)best=imax64(best,bestgap[i]);
    best=imin64(best,unknowngap);// it is possible that best>unknowngap, in this case we should set
                                 // best=unknowngap to see all record gaps.
    fprintf(fout,"Listing gaps>=%d\n",best);
    for(i=0;i<cnt;i++)if(g[i].gap>=best&&(i==0||g[i].p1!=g[i-1].p1))fprintf(fout,"%u %llu\n",g[i].gap,g[i].p1);
    
    fprintf(fout,"Listing the best gaps for each res in [%u,%u):\n",RES1,RES2);
    for(i=0,res=RES1;res<RES2;res++){
        fprintf(fout,"res=%d\n",res);
        for(;i<cnt&&g[i].res<=res;i++)
           if(g[i].res==res&&(g[i].gap>=unknowngap||g[i].gap==bestgap[res])&&(i==0||g[i].p1!=g[i-1].p1))
               fprintf(fout,"%u %llu\n",g[i].gap,g[i].p1);
    }
    fclose(fout);
    printf("See the report in the gap_report.txt file.\n");
            
    free(bestgap);
    free(g);
}

void init_smallp_segment_sieve(bucket *C,ui64 res){

    ui32 i;
    for(i=0;i<ppi;i++){
        C[i].offset=lin_solve(res,primes[i]);
        C[i].pr=primes[i];}
}

void init_segment_sieve(ui64 res,bucket *B,ui32 *previous_bucket,
       ui32 *available_buckets,ui32 *large_lists,ui32 *first_bucket,ui32 *my_head,
       si32 *my_num_available_buckets,ui32 p1,ui32 p2,int thread_id){

   if(p1>p2)return;// no prime

   ui32 tmp32;
   ui32 i,j,o,o2,p,pos,head=0;
   si32 num_available_buckets=0;
   ui32 size2=num_bucket/primes_per_bucket;

   for(i=0;i<size2;i++)available_buckets[i]=i<<sh3;
   num_available_buckets=size2;
   for(i=0;i<num_sieve;i++)large_lists[i]=0;

   ui32 st=((p1>>7)/(8*threads))*8*threads;
   st+=8*thread_id;
   if(st<(p1>>7))st+=8*threads;

   for(i=0;(ui64)i*LEN<=(ui64)p2;i++)if(TH[i]==thread_id||i<=threads){
       ui32 q1=imax64(p1,i*LEN);
       ui32 q2=imin64(p2,i*LEN+(LEN-1));
       ui32 j1=q1>>7;
       ui32 j2=q2>>7;
       ui32 step,k,mk;

       if(i<=threads){j1+=8*thread_id;step=8*threads;mk=8;}
       else          {step=1;mk=1;}

       for(j=j1;j<=j2;j+=step)for(k=0;k<mk&&j+k<=(q2>>7);k++){

       ui64 temp64=isprime_table[j+k];
       for(p=128*(j+k)+1;temp64;temp64>>=1,p+=2){
           int e=get_lsb(temp64);
           temp64>>=e;
            p+=(e<<1);
          if(p<q1||p>q2||(mod%p==0))continue;

           o=lin_solve(res,p);

          o2=o>>sieve_length_bits_log2;
          pos=large_lists[o2];
          if((pos&hash3)==0){
             if(pos==0){
                pos=available_buckets[head];
                large_lists[o2]=pos;
                first_bucket[pos>>sh3]=1;

                head++;if(head==size2)head=0;num_available_buckets--;
             }
             else{
                tmp32=pos-primes_per_bucket;
                pos=available_buckets[head];
                large_lists[o2]=pos;
                previous_bucket[pos>>sh3]=tmp32;
                first_bucket[pos>>sh3]=0;

                head++;if(head==size2)head=0;num_available_buckets--;
             }
          }
          B[pos].pr=p;
          B[pos].offset=o&hash2;
          large_lists[o2]++;
   }}}
   assert(num_available_buckets>=0);
   *my_head=head;
   *my_num_available_buckets=num_available_buckets;

   return;
}

ui64 size_offset;
void sieve_small_primes(ui64 *sieve_array,bucket *C,ui32 num_intervals,
                        ui64 c0,ui32 p1,ui32 p2){

    assert(cnt_offsets>0);
    ui32 cnt2,g,h,i,j,k,o,p,st[cnt_offsets];
    ui64 *arr=sieve_array;

    if(LEN<131072)cnt2=0;
    else          cnt2=PPI_131072;// PPI_131072 number of primes not used in small sieves up to 2^17

    int mult;
    if(USE_AVX2){mult=AVX2_BITS;}
    else        {mult=64;}
    assert((mult&(mult-1))==0);
    
    for(i=0;i<cnt_offsets;i++){
        ui32 P2=PROD[i];
        ui64 inv=single_modinv(mod,P2);
        ui64 pos=mulmod(inv,c0%P2,P2)%P2;
        k=((mult-(pos&(mult-1)))*inv_table[P2&(mult-1)])&(mult-1);// P2 is odd
        pos+=(ui64)k*P2;
        assert(pos%mult==0);
        pos/=mult;
        st[i]=pos;
        if(USE_AVX2)st[i]*=AVX2_BITS/64;
    }

    if(USE_AVX2)assert(LEN64%(AVX2_BITS/64)==0);
    
    for(h=0;h<num_intervals;arr+=LEN64,h++){
        ui64 sh=0;
        for(i=0;i<cnt_offsets;i++){
            ui64 *arr2=arr;
            ui64 *off2=offsets+sh+st[i];
            ui64 add=(ui64)h*LEN64;
            ui64 add2=sh+st[i];

            sh+=SIZE[i]>>6;
            if(i==0){
              #if 1
              memcpy(arr2,off2,8*LEN64);
              #else 
              for(k=0;k+15<LEN64;){
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
                arr2[k]=off2[k];k++;
            }
            for(;k<LEN64;k++)arr2[k]=off2[k];
            #endif
            }
            else{
#if USE_AVX2
               __m256i avx_i1;
		       __m256i avx_i2;
		       __m256i avx_i3;
               __m256i avx_i4;
               __m256i avx_i5;
               __m256i avx_i6;
               __m256i avx_i7;
               __m256i avx_i8;
               __m256i avx_i9;
		       __m256i avx_i10;
		       __m256i avx_i11;
               __m256i avx_i12;
               __m256i avx_i13;
               __m256i avx_i14;
               __m256i avx_i15;
               __m256i avx_i16;

                for(k=0;k+31<LEN64;add+=32,add2+=32,k+=32){
                    avx_i1=_mm256_load_si256((__m256i *)&sieve_array[add]);
                    avx_i2=_mm256_load_si256((__m256i *)&sieve_array[add+4]);
                    avx_i3=_mm256_load_si256((__m256i *)&sieve_array[add+8]);
                    avx_i4=_mm256_load_si256((__m256i *)&sieve_array[add+12]);
                    avx_i5=_mm256_load_si256((__m256i *)&sieve_array[add+16]);
                    avx_i6=_mm256_load_si256((__m256i *)&sieve_array[add+20]);
                    avx_i7=_mm256_load_si256((__m256i *)&sieve_array[add+24]);
                    avx_i8=_mm256_load_si256((__m256i *)&sieve_array[add+28]);

                    avx_i9= _mm256_load_si256((__m256i *)&offsets[add2]);
                    avx_i10=_mm256_load_si256((__m256i *)&offsets[add2+4]);
                    avx_i11=_mm256_load_si256((__m256i *)&offsets[add2+8]);
                    avx_i12=_mm256_load_si256((__m256i *)&offsets[add2+12]);
                    avx_i13=_mm256_load_si256((__m256i *)&offsets[add2+16]);
                    avx_i14=_mm256_load_si256((__m256i *)&offsets[add2+20]);
                    avx_i15=_mm256_load_si256((__m256i *)&offsets[add2+24]);
                    avx_i16=_mm256_load_si256((__m256i *)&offsets[add2+28]);

                    avx_i1=_mm256_and_si256(avx_i1,avx_i9);
                    avx_i2=_mm256_and_si256(avx_i2,avx_i10);
                    avx_i3=_mm256_and_si256(avx_i3,avx_i11);
                    avx_i4=_mm256_and_si256(avx_i4,avx_i12);
                    avx_i5=_mm256_and_si256(avx_i5,avx_i13);
                    avx_i6=_mm256_and_si256(avx_i6,avx_i14);
                    avx_i7=_mm256_and_si256(avx_i7,avx_i15);
                    avx_i8=_mm256_and_si256(avx_i8,avx_i16);

                    _mm256_store_si256((__m256i *)&sieve_array[add],avx_i1);
                    _mm256_store_si256((__m256i *)&sieve_array[add+4],avx_i2);
                    _mm256_store_si256((__m256i *)&sieve_array[add+8],avx_i3);
                    _mm256_store_si256((__m256i *)&sieve_array[add+12],avx_i4);
                    _mm256_store_si256((__m256i *)&sieve_array[add+16],avx_i5);
                    _mm256_store_si256((__m256i *)&sieve_array[add+20],avx_i6);
                    _mm256_store_si256((__m256i *)&sieve_array[add+24],avx_i7);
                    _mm256_store_si256((__m256i *)&sieve_array[add+28],avx_i8);
                }
            for(;k<LEN64;k++)arr2[k]&=off2[k];
#else
            for(k=0;k+15<LEN64;){
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
                arr2[k]&=off2[k];k++;
            }
            for(;k<LEN64;k++)arr2[k]&=off2[k];
#endif
            }
#if USE_AVX2
st[i]=(st[i]+LEN64)%(PROD[i]*(AVX2_BITS/64));
#else
st[i]=(st[i]+LEN64)%PROD[i];
#endif
        }
        ui64 *arr2=arr;
        for(g=0;g<(LEN>>17);arr2+=2048,g++){
            for(j=0;j<PPI_131072;j++){
               o=C[j].offset;
               p=C[j].pr;
               for(k=o;k<131072;k+=p)arr2[k>>6]&=InvBits[k&63];
               C[j].offset=k-131072;
        }}
        for(j=cnt2;j<PPI_LEN;j++){
            o=C[j].offset;
            p=C[j].pr;
            for(k=o;k<LEN;k+=p)arr[k>>6]&=InvBits[k&63];
            C[j].offset=k-LEN;
        }
    }
    
    // correction the offsets for the next block of array
    for(j=0;j<PPI_LEN;j++){
        C[j].offset+=res_table[j];
        if(C[j].offset>=C[j].pr)C[j].offset-=C[j].pr;
    }

    return;
}

void segmented_sieve(ui64 *sieve_array,
                     si64 num_intervals,bucket *B,ui32 *previous_bucket,
                     ui32 *available_buckets,ui32 *large_lists,ui32 *first_bucket,ui32 *my_head,
                     si32 *my_num_available_buckets,ui32 *my_start_offset_bucket){
   if(num_intervals<=0)return;

   ui32 tmp32;
   ui32 size2=num_bucket/primes_per_bucket;
   ui32 k,o,o1,o2,p,pos,offset_bucket;
   ui32 head=*my_head;
   si32 num_available_buckets=*my_num_available_buckets;
   ui64 *arr=sieve_array;
   ui64 iterations;

   assert(num_available_buckets>=0);

   for(iterations=0,offset_bucket=(*my_start_offset_bucket);iterations<num_intervals;arr+=LEN64,
       offset_bucket=(offset_bucket+1)&hash0,iterations++){
       
       o1=large_lists[offset_bucket];

       if(o1!=0){

		   ui32 first_pos=((o1-1)>>sh3)<<sh3;// we really need o1-1
		   ui32 last_pos=o1;
		   ui32 off1=offset_bucket+1;
		   large_lists[offset_bucket]=0;

		   for(;;){
			   for(k=first_pos;k<last_pos;k++){
				   p=B[k].pr;
				   o=B[k].offset;
				   arr[o>>6]&=InvBits[o&63];
					   
				   o+=p-LEN;// o<LEN --> o+(p-LEN)<p<2^32, so there is no overflow in 32 bits! (furthermore p-LEN>0)
				   o2=((o>>sieve_length_bits_log2)+off1)&hash0;
				   pos=large_lists[o2];

				  if((pos&hash3)==0){
					 if(pos==0){
						 pos=available_buckets[head];
						 large_lists[o2]=pos;
						 // no need to set previous_bucket, first_bucket[] will show
						 // that this is the first, there is no previous
						 first_bucket[pos>>sh3]=1;
						 head++;if(head==size2)head=0;num_available_buckets--;
						}
						else{
						 tmp32=pos-primes_per_bucket;
						 pos=available_buckets[head];
						 large_lists[o2]=pos;
						 previous_bucket[pos>>sh3]=tmp32;
						 first_bucket[pos>>sh3]=0;
						 head++;if(head==size2)head=0;num_available_buckets--;
						}
					 if(num_available_buckets<0){
						 printf("Bug, out of run in buckets.\n");
						 exit(1);
						}
					}

				   B[pos].pr=p;
				   B[pos].offset=o&hash2;
				   large_lists[o2]++;
				}
			    o1=(o1-1)>>sh3;
			    o2=previous_bucket[o1];
			    available_buckets[(head+num_available_buckets)%size2]=o1<<sh3;// give back to the list this bucket
			    num_available_buckets++;

			    if(first_bucket[o1])break;// no more bucket for this interval
			    first_pos=o2;
			    last_pos=o2+primes_per_bucket;
			    o1=o2+1;// because there will be (o1-1)>>sh3 we need to do this
            }
		}
   }
   assert(num_available_buckets>=0);

   *my_num_available_buckets=num_available_buckets;
   *my_head=head;
   *my_start_offset_bucket=offset_bucket;

   return;
}

ui32 size_isprime;
ui64 size_primes;
void basic_segmented_sieve(ui32 n){// find primes up to max(n,65536)

    ui32 LEN2=imax64(LEN,65536);
    n=imax64(n,LEN2);
    n-=n%65536;
    n+=65535;

    ui32 i,j,k,m,m2,p,plist[6542],offset[6542];
    ui64 temp64,a[512];

    // First get the primes up to 65536
    for(i=0;i<512;i++)a[i]=inf64;
    a[0]&=InvBits[0];
    for(i=1;i<128;i++)if(a[i>>6]&Bits[i&63]){
        p=(i<<1)+1;
        for(j=(p*p-1)>>1;j<32768;j+=p)a[j>>6]&=InvBits[j&63];
    }

    plist[0]=2;ppi=1;
    for(i=0;i<32768;i++)if(a[i>>6]&Bits[i&63])plist[ppi++]=(i<<1)+1;
    printf("primepi(65536)=%d;\n",ppi);
    assert(ppi==6542);

    size_isprime=mround_512(8*((n>>7)+1));
    if(posix_memalign((void**)&isprime_table,ALIGNEMENT,size_isprime)!=0)print_error_msg();

    for(i=0;i<=(n>>7);i++)isprime_table[i]=inf64;
    isprime_table[0]&=InvBits[0];
    for(i=1;i<ppi;i++)offset[i]=(plist[i]*plist[i]-1)>>1;
    for(i=0;i<=(n>>16);i++){
        m=32768*i+32767;
        m2=(m<<1)+1;
        for(j=1;j<6542;j++){
            p=plist[j];
            if(p*p>m2)break;
            for(k=offset[j];k<=m;k+=p)
                isprime_table[k>>6]&=InvBits[k&63];
            offset[j]=k;
        }
    }

    ui32 cnt=0;
    for(i=0;i<=(LEN2>>7);i++)
       {for(temp64=isprime_table[i];temp64;temp64>>=1)cnt+=temp64&1;}
    cnt=mround_512(cnt);
    ui64 size_primes=(ui64)cnt*sizeof(ui32);
    if(posix_memalign((void**)&primes,ALIGNEMENT,size_primes)!=0)print_error_msg();
    ppi=0;
    for(i=0;i<=(LEN2>>7);i++){
       for(p=128*i+1,temp64=isprime_table[i];temp64;p+=2,temp64>>=1)if(temp64&1){
           if(p<128||p>LEN2||mod%p==0)continue;
           primes[ppi++]=p;
       }
    }
    printf("done basic segment sieve\n");

    return;
}

void get_params(void){

    int ret;
    char u[256],w[256];

    FILE *fin;
    fin=fopen("worktodo_gap.txt","r");

    if(fin!=NULL){
        printf("Found unfinished work, do you want to continue that (y/n) ? ");
        ret=scanf("%s",w);
        if(w[0]=='y'||w[0]=='Y'){
            ret=fscanf(fin,"%s",w);
            double vnum;
            sscanf(w,"version=%lf",&vnum);
            if(w[0]!='v'||vnum<1.0){print_err();exit(1);}

            printf("Read and use n1,n2,n,res1,res2,res,m1,m2 from the worktodo file.\n");
            ret=fscanf(fin,"%s",w);if(1)   {sscanf(w,"n1=%s",u);first_n=conv64(u);set_n1=1;}
            ret=fscanf(fin,"%s",w);if(1)   {sscanf(w,"n2=%s",u);last_n=conv64(u);set_n2=1;}
            ret=fscanf(fin,"%s",w);if(1)   {sscanf(w,"n=%s",u);n0=conv64(u);set_n=1;}
            ret=fscanf(fin,"%s",w);if(1)   {sscanf(w,"res1=%u",&RES1);set_res1=1;}
            ret=fscanf(fin,"%s",w);if(1)   {sscanf(w,"res2=%u",&RES2);set_res2=1;}
            ret=fscanf(fin,"%s",w);if(1)   {sscanf(w,"res=%u",&RES);set_res=1;}
            ret=fscanf(fin,"%s",w);if(1)   {sscanf(w,"m1=%u",&M1);set_m1=1;}
            ret=fscanf(fin,"%s",w);if(1)   {sscanf(w,"m2=%u",&M2);set_m2=1;}
            if(fscanf(fin,"%s",w)!=EOF){// in previous versions there was no saved stepk value
                sscanf(w,"stepk=%llu",&step_k2);
                set_stepk=1;
            }
        }
        // else remove("worktodo_gap.txt"); note: we remove this file later
        fclose(fin);
    }

    if(!set_n1){
        printf("Give the first tested number! ");
        ret=scanf("%s",w);first_n=conv64(w);
        set_n1=1;
        if(first_n<=1+inf32){printf("Too small, n1>2^32 should be true.\n");exit(1);}
    }
    if(!set_n2){
        printf("Give the last tested number!  ");
        ret=scanf("%s",w);last_n=conv64(w);
        set_n2=1;
        if(last_n>=inf64-inf32){printf("Too large, n2<2^64-2^32 should be true.\n");exit(1);}
    }

    if(!set_n){
        printf("Give n (for a first run n=n1, the first tested number) ");
        ret=scanf("%s",w);
        n0=conv64(w);
        set_n=1;
    }
    
    if(!set_res1){
        printf("Give res1 (first tested residue) ");
        ret=scanf("%u",&RES1);
        set_res1=1;
    }

    if(!set_res2){
        printf("Give res2 (first non-tested(!) residue) ");
        ret=scanf("%u",&RES2);
        set_res2=1;
    }
    
    if(!set_res){
        printf("Give res (for a first run res=res1, the first tested residue) ");
        ret=scanf("%u",&RES);
        set_res=1;
    }

    if(!set_m1){
        printf("Give m1 ");
        ret=scanf("%u",&M1);
        set_m1=1;
    }

    if(!set_m2){
        printf("Give m2 ");
        ret=scanf("%u",&M2);
        set_m2=1;
    }
    
    if(set_makereport){make_a_report();printf("We are exiting.\n");exit(0);}

    if(!set_numcoprime){
        printf("Give numcoprime (default=%d) ",default_numcoprime);
        ret=scanf("%d",&NUMCOPRIME);
        set_numcoprime=1;
    }

    if(!set_sb){
        printf("Give the log2 of the total sum of sieving bits (default=%d) ",default_sieve_bits_log2);
        ret=scanf("%d",&sieve_bits_log2);
        set_sb=1;
    }

    if(!set_bs){
        printf("Give the log2 size of one bucket's byte size (default=%d) ",default_bucket_size_log2);
        ret=scanf("%d",&bucket_size_log2);
        set_bs=1;
    }

    if(sieve_bits_log2<16||(sieve_bits_log2-bucket_size_log2)<3){// smart check
// Is the sieve too small or is the bucket size greater than the sieve size
        printf("Do you really want this(?), not swapped sb-bs values?\n");
        printf("sb is log2 of the sieving size in bits.\n");
        printf("bs is log2 of the one bucket size in bytes.\n");
        printf("You can still use Ctrl+C to break from here.\n");
    }

    if(!set_mem){
       printf("Give for this code the available memory in GB (1GB=2^30 bytes),\nthis can be a real number (say 4.35) ");
       ret=scanf("%lf",&max_memory_gigabytes);
       set_mem=1;
    }

    int maxthreads=omp_get_max_threads();
    if(!set_t){
       printf("Found %d thread(s) on this computer,\n",maxthreads);
       printf("give the number of availables thread(s) for this code! ");
       ret=scanf("%d",&threads);
       set_t=1;
    }
    assert(threads>=0&&threads<=maxthreads);
    if(threads==0)exit(1);
    omp_set_num_threads(threads);
    
    mod=(ui64)M1*M2;

    assert(first_n<=last_n);
    
    assert(RES1>=0&&RES2>RES1&&RES2<=M2);
    
    assert(NUMCOPRIME>2);
    
    assert(M1<=unknowngap);// this is still not sufficient condition, we'll need at least 3 coprimes to m values...
    
    assert(NUM_SIEVE_PRIMES%5==0&&NUM_SIEVE_PRIMES>0&&NUM_SIEVE_PRIMES<=165);
    
    assert(RES>=RES1&&RES<RES2);
    
    assert(n0>=first_n&&n0<last_n);
    
    return;
}

int inits(void){// return 1 if all inits is success

    get_params();

    ui32 prime_per_thread[threads];

    ui32 maxp=imax64(isqrt64(last_n),131072);
    sieve_length_bits_log2=bitlen(Bits[sieve_bits_log2]/threads)-1;
    while(sieve_length_bits_log2>6&&sieve_length_bits_log2>=bitlen(maxp))sieve_length_bits_log2--;
    LEN=Bits[sieve_length_bits_log2];

    assert(bitlen(maxp)>=sieve_length_bits_log2+1);// with this maxp>=LEN is true
    num_sieve=Bits[bitlen(maxp)-sieve_length_bits_log2];
    hash0=num_sieve-1;

    primes_per_bucket=precpower2((1<<bucket_size_log2)/(2*size_ui32));
    hash3=primes_per_bucket-1;
    sh3=bitlen(hash3);

    LEN64=LEN>>6;
    hash2=LEN-1;

    ui32 h,i,j,k,p,cnt;
    ui64 I;
    time_t sec=time(NULL);

    assert(sizeof(ui32)>=4);
    assert(sizeof(ui64)==8);
    
    ui64 size=0;int r;
    for(i=1;i<=NUM_SIEVE_PRIMES;i++)size+=primes2[i];
    bitmap=(ui64*)malloc(size*sizeof(ui64));
    for(size=0,i=1;i<=NUM_SIEVE_PRIMES;i++){
        p=primes2[i];
        for(r=0;r<p;r++){
           bitmap[size+r]=inf64;
           ui64 hash=1;
           for(j=0;j<64;j++,hash<<=1)
               if((r+2*j)%p==0)bitmap[size+r]-=hash;
    }size+=p;}
    
    basic_segmented_sieve(imax64(maxp,LEN));

    ui32 size_mod=mround_512(mod)*sizeof(ui32);
    if(posix_memalign((void**)&inv_mod,ALIGNEMENT,size_mod)!=0)print_error_msg();
    for(i=0;i<mod;i++)inv_mod[i]=single_modinv(i,mod);

    ui64 len=0;
    cnt=0;
    int mult;
    if(USE_AVX2)mult=256;
    else        mult=64;
    for(i=0;i<31;){
        if(mod%primes2[i]>0){
          ui32 prod=primes2[i];
          i++;
          for(;i<31&&prod<MAX_SIZE/primes2[i]*8/mult;){
             if(mod%primes2[i]>0)prod*=primes2[i];
             i++;}

          size=mround_512(lcm64(prod,mult)+LEN);
          len+=size;
          PROD[cnt]=prod;
          SIZE[cnt]=size;
          cnt++;
       }
       else i++;
    }
    size_offset=mround_512(len/8);
    if(posix_memalign((void**)&offsets,ALIGNEMENT,size_offset/8*sizeof(ui64))!=0)print_error_msg();
    
    for(I=0;I<(len>>6);I++)offsets[I]=inf64;// Here assume that sizeof(ui64)=8
    ui64 sh=0;
    for(i=0;i<cnt;i++){
        ui64 *offsets3=offsets+sh;
        size=SIZE[i];
        for(j=0;j<31;j++)if(PROD[i]%primes2[j]==0){
            p=primes2[j];
            for(I=0;I<size;I+=p)offsets3[I>>6]&=InvBits[I&63];
        }
        sh+=size/64;
    }
    cnt_offsets=cnt;

    cnt=0;
    PPI_131072=0;
    PPI_LEN=0;
    for(i=0;i<=(maxp>>7);i++){
        ui64 temp64=isprime_table[i];
        for(p=128*i+1;temp64;temp64>>=1,p+=2){
            int e=get_lsb(temp64);
            temp64>>=e;
            p+=(e<<1);
            if(p>128&&mod%p>0){
               if(p<131072)PPI_131072++;
               if(p<LEN)  PPI_LEN++;
            }
        }
    }

    // Distribute the primes in [LEN,maxp] in such a way that in each thread sum(1/p) will be roughly equal
    //    to get roughly equal time in (the bucket version) segmented sieve
    //    and for a multi threaded run the larger primes will be in blocks in the threads:
    //        giving that (o+p)/LEN will evaluate as a very few different values
    //    it looks like that it will balance also the number of primes in each thread
    
    
    TH=(ui32*)malloc((maxp/LEN+1)*sizeof(ui32));
    double pr[threads];
    for(i=0;i<threads;i++){pr[i]=0.0;prime_per_thread[i]=0;}
    for(i=1;(ui64)i*LEN<=(ui64)maxp;i++){
        if(i<=threads){
            ui32 p1=LEN*i;
            ui32 p2=p1+(LEN-1);
            ui32 j1=p1>>7;
            ui32 j2=p2>>7;
            for(k=0;k<threads;k++)for(j=j1+8*k;j<=j2&&j<=(maxp>>7);j+=8*threads)for(h=0;h<8&&h+j<=(maxp>>7);h++){
                ui64 tmp64=isprime_table[h+j];
                p=128*(h+j)+1;
                for(;tmp64;tmp64>>=1,p+=2){
                    int e=get_lsb(tmp64);
                    tmp64>>=e;
                    p+=(e<<1);
                    if(p<128||p>maxp||mod%p==0)continue;
                    pr[k]+=(double)1/p;
                    prime_per_thread[k]++;
                }
            }
        }
        else{
            ui32 p1=LEN*i,pos=0;
            double low_pr=0.0;
            double s=0.0;
            cnt=0;
            for(j=1;j<LEN;j+=2){
               p=p1+j;
               pos=(p-1)>>1;
               if(p<128||p>maxp||mod%p==0||(isprime_table[pos>>6]&Bits[pos&63])==0)continue;
               cnt++;
               s+=(double)1/p;
            }
            for(j=0;j<threads;j++)
                if(j==0||pr[j]<low_pr){pos=j;low_pr=pr[j];}
            pr[pos]+=s;
            TH[i]=pos;
            prime_per_thread[pos]+=cnt;
        }
    }
    ui32 np=0;
    for(i=0;i<threads;i++)np+=prime_per_thread[i];

    num_bucket=1;
    for(i=0;i<threads;i++)num_bucket=imax64(num_bucket,prime_per_thread[i]);
    num_bucket+=(num_sieve+3)*primes_per_bucket;// it is enough +1 for rounding and +1 for the current bucket
                                                // +1 for a new bucket
    num_bucket=mround_gen(num_bucket,lcm64(512,primes_per_bucket));
    printf("All initializations are done in %ld seconds.\n",time(NULL)-sec);

    return 1;
}

void ff(void){

    ui32 res,maxp=isqrt64(last_n);

    assert(is_power_two(num_sieve)==1);
    assert(sieve_length_bits_log2>=6&&sieve_length_bits_log2<32);

	int threads2 = threads * 2;
    ui32 np=num_bucket/primes_per_bucket;
    ui64 size=(ui64)threads*num_bucket;
    ui64 size2=(ui64)threads*num_sieve;
    ui64 size3=(ui64)threads*((ui64)count_LEN_intervals*(2*LEN/8));
    ui64 size6=(ui64)threads*np;
    ui64 size8=mround_512(ppi);
    ui64 size9=(ui64)threads*mround_512(ppi);

    double m0=0.0,m1=0.0;
    m0+=(double)size_offset;// size of offsets array
    m0+=(double)size_isprime;// size of the isprime array
    m0+=(double)size_primes;// size of the primes array
    m0+=(double)size8*sizeof(ui32);// size of the res_table array
    m0+=(double)mod*sizeof(ui32);// size of inv_mod array

    m1+=(double)num_bucket*sizeof(bucket);// B
    m1+=(double)np*sizeof(ui32);// available_buckets
    m1+=(double)num_sieve*sizeof(ui32);// large_lists
    m1+=(double)count_LEN_intervals*(2*LEN/8);// sieve_array
    m1+=(double)np*sizeof(ui32);// previous bucket
    m1+=(double)np*sizeof(ui32);// first bucket
    m1+=(double)mround_512(ppi)*sizeof(bucket);// C

    m1*=(double)threads;

    if(!set_n){n0=first_n;set_n=1;}
    if(!set_res){RES=RES1;set_res=1;}
    ui64 first_k=n0/mod;
    ui64 last_k=last_n/mod;

    double mbytes=(double)DP30*max_memory_gigabytes;
    mbytes-=m0+m1;
    if(save_nextprimetest)mbytes/=3.0*((double)LEN/8);
    else                  mbytes/=(double)LEN/8;
    mbytes-=0.5;
    si64 si_temp=(si64)mbytes;
    ui64 ilow=count_LEN_intervals;

    if(si_temp<1||si_temp<ilow){
       m0+=(double)ilow*(LEN/8);
       if(save_nextprimetest)m0+=2.0*((double)ilow*(LEN/8));
       m0/=DP30;m0+=0.005;
       m1/=DP30;m1+=0.005;
       printf("You have given too few memory, need at least %.2lf GB of memory. \n",m0+m1);
       exit(1);
    }
    ui64 s0=last_k-first_k+3;
    ui64 num_iterations=mround_gen(imin64((ui64)si_temp,s0/LEN+1),count_LEN_intervals);

    ui64 size4=(ui64)num_iterations*(LEN/8);// size of ans, saved_ans, saved_ans2
    m0+=(double)size4;
    printf("Scanning at once %llu n values (%u consecutive values in each interval length=%llu)\n",
           8*((ui64)size4*M1),M1,mod);
    if(save_nextprimetest)m0+=2.0*((double)size4);
    m0/=DP30;
    m1/=DP30;
    used_memory=m0+m1;
    printf("Memory usage: %.2lf GB.\n",used_memory);

    ui64 *ans,*saved_ans,*saved_ans2,*sieve_array;
    ui32 *large_lists,*previous_bucket,*available_buckets,*first_bucket;
    bucket *B,*C;
    if(posix_memalign((void**)&B,ALIGNEMENT,size*sizeof(bucket))!=0)print_error_msg();
    if(posix_memalign((void**)&C,ALIGNEMENT,size9*sizeof(bucket))!=0)print_error_msg();

    if(posix_memalign((void**)&available_buckets,ALIGNEMENT,size6*sizeof(ui32))!=0)print_error_msg();
    if(posix_memalign((void**)&large_lists,ALIGNEMENT,size2*sizeof(ui32))!=0)print_error_msg();
    if(posix_memalign((void**)&sieve_array,ALIGNEMENT,size3)!=0)print_error_msg();
    if(posix_memalign((void**)&ans,ALIGNEMENT,size4)!=0)print_error_msg();
    if(save_nextprimetest){if(posix_memalign((void**)&saved_ans,ALIGNEMENT,size4)!=0)print_error_msg();}
    else saved_ans=(ui64*)malloc(1*sizeof(ui64));
    if(save_nextprimetest){if(posix_memalign((void**)&saved_ans2,ALIGNEMENT,size4)!=0)print_error_msg();}
    else saved_ans2=(ui64*)malloc(1*sizeof(ui64));
    if(posix_memalign((void**)&previous_bucket,ALIGNEMENT,size6*sizeof(ui32))!=0)print_error_msg();
    if(posix_memalign((void**)&first_bucket,ALIGNEMENT,size6*sizeof(ui32))!=0)print_error_msg();
    if(posix_memalign((void**)&res_table,ALIGNEMENT,size8*sizeof(ui32))!=0)print_error_msg();

    
    int i,j;

    ui64 large_block=(ui64)count_LEN_intervals*LEN;

    printf("Program version number=%s;\n",version);
    printf("USE_AVX2=%d\n",USE_AVX2);
    printf("Start the main algorithm; date: ");print_time();
    printf("\ninterval=[%llu,%llu]; now at n=%llu;res=%u;\n",first_n,last_n,n0,RES);
    printf("test for res in [%u,%u);\n",RES1,RES2);
    printf("m1=%u;m2=%u; (m=%llu);\n",M1,M2,(ui64)M1*M2);
    printf("numcoprime=%d;sb=%d;bs=%d;t=%d threads;memory=%.2lf GB\n",
           NUMCOPRIME,sieve_bits_log2,bucket_size_log2,threads,used_memory);

    ui64 step_k=(ui64)num_iterations*LEN;// num_iterations is divisible by count_LEN_intervals
    ui32 ppi2=mround_512(ppi);
    
    
    // choose the correct step value for k
    ui64 tmp64=step_k;
    if(!set_stepk){step_k2=step_k;set_stepk=1;}
    step_k=mround_gen(imin64(step_k,(ui64)LEN*(step_k2/LEN)),LEN);//note it is possible that step_k2 is not divisible by LEN
                                                    // with this step_k<=step_k2,previous step_k value
    assert(step_k>0);
    num_iterations=step_k/LEN;// here step_k is divisible by LEN
    step_k2=tmp64;

    
    ui64 temp=(ui64)large_block*(threads-1);
    for(i=0;i<ppi;i++){
        ui32 tmp2=primes[i]-(temp%primes[i]);
        if(tmp2==primes[i])tmp2=0;
        res_table[i]=tmp2;
    }
    
    int mingap=imin64(unknowngap,default_report_gap);
    int saved,saved2;
    int parity=0;
    ui32 mid_res=0,mid_res2=0;
    ui64 processed_k=0;
    FILE* fout;
    
    time_t sec=time(NULL);
    int first_iteration=1;
    for(;first_k<=last_k;first_k+=step_k){
       if(!first_iteration){
           step_k=step_k2;
           num_iterations=step_k/LEN;// step_k is divisible by LEN
       }
       first_iteration=0;
       
       
       saved=0;
       saved2=0;
       for(;RES<RES2;RES++){

       num_coprime=0;
       ui64 x0=(ui64)RES*M1;
       for(num_coprime=0,i=0;num_coprime<NUMCOPRIME&&M1+i<=unknowngap;i++)
          if(gcd64(x0+i,mod)==1)r_table[num_coprime++]=i;// note: we have to ensure that m1+gap_delta<=unknowngap
                                                     // this means that it is possible that numcoprime<NUMCOPRIME
                                                     // if this happens a lot of times then it means that
                                                     // M1 is far from optimal choice...
       assert(num_coprime>2);
       gap_delta=r_table[num_coprime-1];
       ui64 nn=imax64(first_n,first_k*mod);
       printf("   Test: n=%llu;res=%u; (delta=%d;numcoprime=%d;)\n",nn,RES,gap_delta,num_coprime);
       
       fout=fopen("gap_log.txt","a+");
       fprintf(fout,"   Test: n=%llu;res=%u; (delta=%d);\n",nn,RES,gap_delta);
       fclose(fout);
       
       gap_delta=r_table[num_coprime-1]+1;
       while(gcd64(x0+gap_delta,mod)>1)gap_delta++;

        fout=fopen("worktodo_gap.txt","w");
        fprintf(fout,"version=%s\n",version);
        fprintf(fout,"n1=%llu\n",first_n);
        fprintf(fout,"n2=%llu\n",last_n);
        fprintf(fout,"n=%llu\n",nn);
        fprintf(fout,"res1=%u\n",RES1);
        fprintf(fout,"res2=%u\n",RES2);
        fprintf(fout,"res=%u\n",RES);
        fprintf(fout,"m1=%u\n",M1);
        fprintf(fout,"m2=%u\n",M2);
        fprintf(fout,"stepk=%llu\n",step_k);
        fclose(fout);
        
        ui64 num_large_block=imin64(num_iterations/count_LEN_intervals,
                mround_gen(last_k-first_k+1,large_block)/large_block);
        ui64 I2=((ui64)num_large_block*large_block)/64;

        memset(ans,0,(ui64)8*I2);

        temp=imin64(first_k+step_k,last_k);
        temp=imin64(temp*mod,last_n);
        maxp=isqrt64(temp);

		printf("Sieving\r");
		fflush(stdout);
        for(i=num_coprime-1;i>=0;i--){
            if(save_nextprimetest&&i==(num_coprime/2)){
               ui64 I2=((ui64)num_iterations*LEN)/64;
               if(parity)memcpy(saved_ans, ans,(ui64)8*I2);
               else      memcpy(saved_ans2,ans,(ui64)8*I2);
               parity^=1;
                
               saved=saved2;
               saved2=1;
               mid_res=mid_res2;
               mid_res2=r_table[i+1];
            }

            res=r_table[i];
            ui32 head[threads];
            ui32 start_offset_bucket[threads];
            si32 num_available_buckets[threads];

            ui64 it;
			ui64 lb64=large_block/64;
            ui64 c1=first_k*mod+((ui64)RES*M1);c1+=res;
            for(it=0;it<num_large_block+threads-1;it++){
                #pragma omp parallel for schedule(dynamic,1)
                for(j=0;j<threads;j++){
                ui32 id2=(it+threads2-j)%threads2;
                ui32 id=(id2+threads)%threads2;

                if(id==threads)id=0;
                else if(id==0)id=threads;
				
				ui64 jnp = (ui64)j*np;
                bucket *B2=B+(ui64)j*num_bucket;
                bucket *C2=C+(ui64)(id2%threads)*ppi2;
                ui32 *previous_bucket2=previous_bucket+jnp;
                ui32 *available_buckets2=available_buckets+jnp;
                ui32 *large_lists2=large_lists+(ui64)j*num_sieve;
                ui32 *first_bucket2=first_bucket+jnp;
                ui64 *sieve_array2=sieve_array+(ui64)id2*lb64;
                ui64 *sieve_array3=sieve_array+(ui64)id*lb64;

                ui64 c0=(ui64)(it+(id%threads))*large_block;
                c0=c1+c0*mod;
                
                if(it==0){// init the sieves
                    init_smallp_segment_sieve(C2,c0);

                    start_offset_bucket[j]=0;
                    head[j]=0;
                    num_available_buckets[j]=0;
                    init_segment_sieve(c1,B2,previous_bucket2,
                       available_buckets2,large_lists2,first_bucket2,&head[j],
                       &num_available_buckets[j],LEN,maxp,j);
                }

                if(it%threads==0&&it<num_large_block){// do the smallprime sieve
                   sieve_small_primes(sieve_array3,C2,count_LEN_intervals,
                   c0,128,LEN);
                }

                // do the (it-j)-th block
                if(it>=j&&it-j<num_large_block){
                    segmented_sieve(sieve_array2,
                       count_LEN_intervals,B2,previous_bucket2,
                       available_buckets2,large_lists2,first_bucket2,&head[j],
                       &num_available_buckets[j],&start_offset_bucket[j]);
                }
                    
                int tst=0,id3;
                if(it+1==num_large_block+threads-1){
                    int res2=(num_large_block+threads-1)%threads+1;
                    if(j==threads-1){tst=1;id3=num_large_block-1;}
                    else if(j+1<res2){tst=1;id3=num_large_block-2-j;}
                }
                else if((it+2)%threads==0&&it+2>=2*threads){
                    tst=1;
                    if(j==threads-1)id3=it-j;
                    else            id3=it-j-threads;         
                }
                if(tst){//save the large block
                       ui64 K,K2=lb64;
                       ui64 *ans2=ans+(ui64)K2*id3;
                       ui64 *sieve_array4=sieve_array+(ui64)(id3%(2*threads))*K2;
                       for(K=0;K<K2;K++)ans2[K]|=sieve_array4[K];
                }
                }}
        }

		printf("Searching\r");
		fflush(stdout);
        int nt=threads;
        ui64 *sols,K;
        ui32 num_solutions[nt];
        sols=(ui64*)malloc(MNS2*nt*sizeof(ui64));
        for(i=0;i<nt;i++){
		num_solutions[i]=0;}

        #pragma omp parallel for schedule(dynamic,1)
        for(K=0;K<I2;K+=1048576){
            int th_id=omp_get_thread_num();
            ui32 h;
            ui64 G,k2=imin64(K+1048576,I2);
            ui64 range_k=k2-K;
            ui64 *ans2=ans+K;
               for(G=0;G<range_k;G++)if(ans2[G]!=inf64){
                   ui64 tmp64=~ans2[G];
                   for(h=0;tmp64;tmp64>>=1,h++){// so now hunting for bit=1
					   int shift = __builtin_ctzll(tmp64);
					   h+=shift;
					   tmp64>>=shift;
                       ui64 pos=64*(K+G)+h;

                       ui64 mult=first_k+pos;
                       if(mult<=last_k){
                          ui64 n=(ui64)mult*mod;
                          n+=(ui64)RES*M1;
                          // there is no prime in [n,n+gap_delta)
                          ui64 p2=next_prime(n+gap_delta),p1;
                          if(saved&&(\
                              (parity&&(saved_ans[pos>>6]&Bits[pos&63]))||\
                              (parity==0&&(saved_ans2[pos>>6]&Bits[pos&63])))){
                              p1=n-M1+mid_res;
                              if(p2-p1<unknowngap)continue;
                          }
                          p1=prec_prime(n-1);
                          if(p2-p1>=mingap){
                             ui32 o=MNS2*th_id+2*num_solutions[th_id];
                             sols[o]=p2-p1;
                             sols[o+1]=p1;
                             num_solutions[th_id]++;
                             if(num_solutions[th_id]==MAX_NUM_SOLUTIONS){
                                 ui32 k;
                                 o=MNS2*th_id;
                                 FILE* fout;
                                 fout=fopen("gap_solutions.txt","a+");
                                 for(k=0;k<num_solutions[th_id];k++){
                                     printf("%llu %llu\n",sols[o+2*k],sols[o+2*k+1]);
                                     fprintf(fout,"%llu %llu\n",sols[o+2*k],sols[o+2*k+1]);
                                 }
                                 fclose(fout);
                                 num_solutions[th_id]=0;
                             }
                          }
                       }
                   }
               }
            }

		ui32 ns=0;for(i=0;i<nt;i++)ns+=num_solutions[i];
        GAP my_gap[ns];

//		Sort the results as they are gathered from separate threads
        ns=0;
        for(j=0;j<nt;j++){
            ui32 k,o=MNS2*j;
            for(k=0;k<num_solutions[j];k++){
				ui32 jj,k2 = o + (k<<1);
				GAP sort;
                sort.gap=(ui32)sols[k2];
                sort.p1=sols[k2+1];
				for (jj = ns; jj > 0 && my_gap[jj-1].p1 > sort.p1; jj--) my_gap[jj] = my_gap[jj-1];
				my_gap[jj] = sort;
                ns++;
            }
        }

        FILE* fout;
        fout=fopen("gap_solutions.txt","a+");
        for(i=0;i<ns;i++)if(i==0||my_gap[i].p1>my_gap[i-1].p1){
            printf("%u %llu\n",my_gap[i].gap,my_gap[i].p1);
            fprintf(fout,"%u %llu\n",my_gap[i].gap,my_gap[i].p1);
        }
        fclose(fout);
        free(sols);

        ui64 k2=imin64(last_k+1,first_k+(ui64)num_large_block*large_block);
        nn=imin64(k2*mod,last_n);
        processed_k+=k2+1-first_k;
        double rate=((double)processed_k*M1)/((double)(time(NULL)-sec)+0.001);// to avoid division by zero
        printf("   %.2lfe9 n/sec.; time=%ld sec.; date: ",rate/1e9,time(NULL)-sec);
        print_time();
        
        ui64 num_n=last_k+1-first_k;
        num_n*=RES2-RES1;
        num_n-=(RES-RES1+1)*(k2+1-first_k);
		double eta=((double)num_n*M1)/rate;
        if(RES==RES2-1&&first_k+step_k>last_k)eta=0.0;// exact, there is no more k value.
        printf("; ETA: %.2lfhrs\n",eta/3600.0);
        fflush(stdout);

        fout=fopen("gap_log.txt","a+");        
        fprintf(fout,"Done interval=[%llu,%llu]; m1=%u;m2=%u; res=%u with version=%s;numcorpime=%d;sb=%d;bs=%d;t=%d threads;memory=%.2lf GB.\n",
            imax64(first_n,(ui64)first_k*mod),nn,M1,M2,RES,version,NUMCOPRIME,sieve_bits_log2,bucket_size_log2,threads,used_memory);
        fclose(fout);
    }
    RES=RES1;}
    
    fout=fopen("results_gap.txt","a+");
    fprintf(fout,"Done interval=[%llu,%llu]; m1=%u;m2=%u; res in [%u,%u) with version=%s;numcoprime=%d;sb=%d;bs=%d;t=%d threads;memory=%.2lf GB.\n",
            first_n,last_n,M1,M2,RES1,RES2,version,NUMCOPRIME,sieve_bits_log2,bucket_size_log2,threads,used_memory);
    fclose(fout);
    remove("worktodo_gap.txt");

    free(ans);
    free(saved_ans);
    free(saved_ans2);
    free(sieve_array);
    free(B);
    free(available_buckets);
    free(previous_bucket);
    free(large_lists);
    free(first_bucket);
    free(offsets);
    free(inv_mod);
    free(primes);
    free(isprime_table);
    free(TH);
    free(C);
    free(res_table);
    
    make_a_report();

    return;
}


int main(int argc, char **argv){

    
    unknowngap=default_unknowngap;
    set_n1=0;
    set_n2=0;
    set_n=0;
    set_res1=0;
    set_res2=0;
    set_res=0;
    set_m1=0;
    set_m2=0;
    set_numcoprime=0;
    set_sb=0;
    set_bs=0;
    set_mem=0;
    set_t=0;
    set_stepk=0;
    set_makereport=0;
    
    while((argc>1)&&(argv[1][0]=='-')){
      if(strcmp(argv[1],"-h")==0||strcmp(argv[1],"--help")==0){
        usage();
        exit(1);
      }
      else if(argc>2&&strcmp(argv[1],"-n1")==0){
	    first_n=conv64(argv[2]);
	    set_n1=1;
	    argv+=2;
	    argc-=2;
	  }
      else if(argc>2&&strcmp(argv[1],"-n2")==0){
	    last_n=conv64(argv[2]);
	    set_n2=1;
	    argv+=2;
	    argc-=2;
	  }
      else if(argc>2&&strcmp(argv[1],"-n")==0){
        n0=conv64(argv[2]);
	    set_n=1;
	    argv+=2;
	    argc-=2;
	  }
      else if(argc>2&&strcmp(argv[1],"-res1")==0){
        RES1=atoi(argv[2]);
	    set_res1=1;
	    argv+=2;
	    argc-=2;
	  }
      else if(argc>2&&strcmp(argv[1],"-res2")==0){
	    RES2=atoi(argv[2]);
	    set_res2=1;
	    argv+=2;
	    argc-=2;
	  }
      else if(argc>2&&strcmp(argv[1],"-res")==0){
	    RES=atoi(argv[2]);
	    set_res=1;
	    argv+=2;
	    argc-=2;
	  }
	  else if(argc>2&&strcmp(argv[1],"-m1")==0){
        M1=atoi(argv[2]);
	    set_m1=1;
	    argv+=2;
	    argc-=2;
	  }
      else if(argc>2&&strcmp(argv[1],"-m2")==0){
	    M2=atoi(argv[2]);
	    set_m2=1;
	    argv+=2;
	    argc-=2;
	  }
	  else if(argc>2&&strcmp(argv[1],"-numcoprime")==0){
        NUMCOPRIME=atoi(argv[2]);
        set_numcoprime=1;
        argv+=2;
        argc-=2;
      }
	  else if(argc>2&&strcmp(argv[1],"-sb")==0){
        sieve_bits_log2=atoi(argv[2]);
        set_sb=1;
        argv+=2;
        argc-=2;
      }
      else if(argc>2&&strcmp(argv[1],"-bs")==0){
        bucket_size_log2=atoi(argv[2]);
        set_bs=1;
        argv+=2;
        argc-=2;
      }
      else if(argc>2&&(strcmp(argv[1],"-mem")==0||strcmp(argv[1],"-memory")==0)){// also accept the -memory switch
        max_memory_gigabytes=atof(argv[2]);// convert to double
        set_mem=1;
        argv+=2;
        argc-=2;
      }
      else if(argc>2&&strcmp(argv[1],"-t")==0){
        threads=atoi(argv[2]);
        set_t=1;
        argv+=2;
        argc-=2;
      }
      else if(argc>2&&strcmp(argv[1],"-unknowngap")==0){
        unknowngap=atoi(argv[2]);
        argv+=2;
        argc-=2;
      }
      else if(argc>1&&strcmp(argv[1],"-makereport")==0){
        set_makereport=1;
        argv+=1;
        argc-=1;
      }
      else{
	    fprintf(stderr,"Unknown option: %s\n",argv[1]);
	    exit(1);
	  }
    }
    
    
    inits();
    time_t sec=time(NULL);
    double dt=cpu_time();
    ff();
    printf("Done interval=[%llu,%llu]; m1=%u;m2=%u; res in [%u,%u) with version=%s;numcoprime=%d;sb=%d;bs=%d;t=%d threads;memory=%.2lf GB.\n",
            first_n,last_n,M1,M2,RES1,RES2,version,NUMCOPRIME,sieve_bits_log2,bucket_size_log2,threads,used_memory);
    printf("Search used %ld sec. (Wall clock time), %.2lf cpu sec.\n",time(NULL)-sec,cpu_time()-dt);
    print_time();printf("\n");

    return 0;
}
