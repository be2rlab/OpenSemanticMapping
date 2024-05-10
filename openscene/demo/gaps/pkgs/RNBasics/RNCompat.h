/* Include file for machine dependent switches */
#ifndef __RN__COMPAT__H__
#define __RN__COMPAT__H__



/************************************************************************* 
Compile Switches:
*************************************************************************/

#define RN_NULL 0



/* Operating system selection */

#define RN_IRIX 1
#define RN_WINDOWS 2
#define RN_LINUX 3
#define RN_MAC 4

#ifdef _WIN32
#  define RN_OS RN_WINDOWS
#else 
#  ifdef __APPLE__
#      define RN_OS RN_MAC
#  else 
#      ifdef RN_USE_SGI
#          define RN_OS RN_IRIX
#      else 
#          define RN_OS RN_LINUX
#      endif
#  endif
#endif



/* Compiler selection */

#define RN_CFRONT 1
#define RN_MSVC 2
#define RN_NCC 3
#define RN_GCC 4

#ifdef _MSC_VER
#  define RN_CC RN_MSVC
#else 
#  ifdef __GNUC__
#      define RN_CC RN_GCC
#  else 
#      ifdef RN_USE_SGI
#          define RN_CC RN_NCC
#      else 
#          define RN_CC RN_GCC
#      endif
#  endif
#endif



/* Language standard */

#define RN_C99 1
#define RN_C11 2

#if (__cplusplus == 201103L)
#   define RN_CC_VER RN_C11
#else
#   define RN_CC_VER RN_C99
#endif



/* Graphics library selection */

#define RN_OPENGL 1
#define RN_XLIB 2
#ifdef RN_USE_XLIB
#   define RN_2D_GRFX RN_XLIB
#   define RN_3D_GRFX RN_OPENGL
#else
#   ifdef RN_USE_OPENGL
#       define RN_2D_GRFX RN_OPENGL
#       define RN_3D_GRFX RN_OPENGL
#   else
#       define RN_2D_GRFX RN_OPENGL
#       define RN_3D_GRFX RN_OPENGL
#   endif
#endif



/* Math precision selection */

#define RN_FLOAT_PRECISION 1
#define RN_DOUBLE_PRECISION 2
#ifdef RN_USE_SINGLE_PRECISION
#   define RN_MATH_PRECISION RN_FLOAT_PRECISION
#else
#   define RN_MATH_PRECISION RN_DOUBLE_PRECISION
#endif



/************************************************************************* 
Compatability definitions
*************************************************************************/

/* Compiler dependent flags */

#if ((RN_OS == RN_IRIX) && (RN_CC == RN_CFRONT))
#   define _SVR4_SOURCE
#   define _SGI_SOURCE
#   define _BSD_COMPAT
#endif

#if (RN_CC == RN_GCC)
#   pragma GCC diagnostic ignored "-Wunused-result"
#endif

#if 0
#if (RN_CC == RN_MSVC)
#   pragma warning(disable : 4244) // Cast of double literals to float
#   pragma warning(disable : 4305) // Cast of double literals to float
#   define _WIN32_WINNT 0x400  // Include newer windows sockets header files
#endif
#endif

#if (RN_OS != RN_WINDOWS)
#  include <stdint.h>
#endif

/* This is needed to avoid error in compiling glu.h in some installations of cygwin */
#ifdef __CYGWIN__
# ifndef CALLBACK
#   if defined(_ARM_)
#     define CALLBACK
#   else
#     define CALLBACK __stdcall
#   endif
# endif
#endif



/************************************************************************* 
Basic type definitions
*************************************************************************/

namespace gaps {
#if (RN_CC == RN_MSVC)
  typedef char                RNChar8; 
  typedef unsigned char       RNUChar8; 
  typedef short               RNInt16; 
  typedef unsigned short      RNUInt16; 
  typedef long                RNInt32; 
  typedef unsigned long       RNUInt32; 
  typedef long long           RNInt64; 
  typedef unsigned long long  RNUInt64; 
  typedef float               RNScalar32; 
  typedef double              RNScalar64;
#else
  typedef int8_t              RNChar8;
  typedef uint8_t             RNUChar8;
  typedef int16_t             RNInt16;
  typedef uint16_t            RNUInt16;
  typedef int32_t             RNInt32;
  typedef uint32_t            RNUInt32;
  typedef int64_t             RNInt64;
  typedef uint64_t            RNUInt64;
  typedef float               RNScalar32;
  typedef double              RNScalar64;
#endif
}



// End include guard
#endif

