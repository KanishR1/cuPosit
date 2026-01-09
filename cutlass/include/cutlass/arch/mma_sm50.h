/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Matrix multiply
*/

#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/complex.h"
#include "cutlass/quaternion.h"
#include "cutlass/functional.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////
// https://maknee.github.io/blog/2025/Maybe-Consider-Putting-Cutlass-In-Your-CUDA-Kernels/
CUTLASS_DEVICE uint32_t cutlass_lutmap(uint32_t input) {
    uint32_t offset = 126 + (input >= 127);
    input -= offset;
    uint32_t result = CUPOSIT_NMANTISSA_MAX - (input >> 2);
    return result & -((int32_t)result < 0);  // returns 0 if negative, result otherwise
}

CUTLASS_DEVICE float posit_clip(float number) {
    // uint32_t bitmask = (0x807FFFFF & (0xFFFFFFFF << (23 - nmantissa)));
    // uint32_t xbits = __float_as_uint(x);
    // xbits = (xbits & bitmask) & (((uint32_t) exp) << 23);
    // return __uint_as_float(xbits);

    // overwrites exponent and truncates mantissa
    // to really round mantissa, it should be incremented when the
    // msb of cutoff bits is 1. but we aren't doing that here.

    // posit_* variables are available in constant memory

    uint32_t x_exponent = (__float_as_uint(number) >> 23) & 0xFF; // see if frexpf is faster
    x_exponent = min( max(x_exponent, CUPOSIT_EXP_MIN), CUPOSIT_EXP_MAX);

    const uint32_t nmantissa_for_exp = cutlass_lutmap(x_exponent);

    return __uint_as_float(
        (__float_as_uint(number) & (0x807FFFFF & (0xFFFFFFFF << (23 - nmantissa_for_exp)))) |
        ((x_exponent) << 23)
    );
} 

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, float, LayoutA, float, LayoutB, float, LayoutC, OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAdd;
  using ElementC = float;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<float, 1> &d,
    Array<float, 1> const &a,
    Array<float, 1> const &b,
    Array<float, 1> const &c
  ) {

#if defined(__CUDA_ARCH__)
    // posit_n will be the same value for every single thread, 
    // so there's no SIMT divergence here, despite the branch
    d[0] = (CUPOSIT_ENABLED == 0) ? (a[0] * b[0] + c[0]) : posit_clip(  posit_clip(a[0] * b[0]) + c[0]);
#else
    d[0] = a[0] * b[0] + c[0];
#endif

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, double, LayoutA, double, LayoutB, double, LayoutC, OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAdd;
  using ElementC = double;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<double, 1> &d,
    Array<double, 1> const &a,
    Array<double, 1> const &b,
    Array<double, 1> const &c
  ) {

    d[0] = a[0] * b[0] + c[0];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, int, LayoutA, int, LayoutB, int, LayoutC, OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAdd;
  using ElementC = int;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<int, 1> &d,
    Array<int, 1> const &a,
    Array<int, 1> const &b,
    Array<int, 1> const &c
  ) {

    d[0] = a[0] * b[0] + c[0];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<
  gemm::GemmShape<1, 1, 1>,
  1,
  complex<float>,
  LayoutA,
  complex<float>,
  LayoutB,
  complex<float>,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAddComplex;
  using ElementC = complex<float>;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<complex<float>, 1> &d,
    Array<complex<float>, 1> const &a,
    Array<complex<float>, 1> const &b,
    Array<complex<float>, 1> const &c
  ) {

    d[0].real() = a[0].real() * b[0].real() + c[0].real();
    d[0].imag() = a[0].imag() * b[0].real() + c[0].imag();
    d[0].real() = -a[0].imag() * b[0].imag() + d[0].real();
    d[0].imag() = a[0].real() * b[0].imag() + d[0].imag();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<
  gemm::GemmShape<1, 1, 1>,
  1,
  complex<float>,
  LayoutA,
  float,
  LayoutB,
  complex<float>,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAddComplex;
  using ElementC = complex<float>;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<complex<float>, 1> &d,
    Array<complex<float>, 1> const &a,
    Array<float, 1> const &b,
    Array<complex<float>, 1> const &c
  ) {

    d[0].real() = a[0].real() * b[0] + c[0].real();
    d[0].imag() = a[0].imag() * b[0] + c[0].imag();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<
  gemm::GemmShape<1, 1, 1>,
  1,
  float,
  LayoutA,
  complex<float>,
  LayoutB,
  complex<float>,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAddComplex;
  using ElementC = complex<float>;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<complex<float>, 1> &d,
    Array<float, 1> const &a,
    Array<complex<float>, 1> const &b,
    Array<complex<float>, 1> const &c
  ) {

    d[0].real() = a[0] * b[0].real() + c[0].real();
    d[0].imag() = a[0] * b[0].imag() + d[0].imag();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<
  gemm::GemmShape<1, 1, 1>,
  1,
  complex<double>,
  LayoutA,
  complex<double>,
  LayoutB,
  complex<double>,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAddComplex;
  using ElementC = complex<double>;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<complex<double>, 1> &d,
    Array<complex<double>, 1> const &a,
    Array<complex<double>, 1> const &b,
    Array<complex<double>, 1> const &c
  ) {

    d[0].real() = a[0].real() * b[0].real() + c[0].real();
    d[0].imag() = a[0].imag() * b[0].real() + c[0].imag();
    d[0].real() = -a[0].imag() * b[0].imag() + d[0].real();
    d[0].imag() = a[0].real() * b[0].imag() + d[0].imag();
  }
};

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<
  gemm::GemmShape<1, 1, 1>,
  1,
  complex<double>,
  LayoutA,
  double,
  LayoutB,
  complex<double>,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAddComplex;
  using ElementC = complex<double>;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<complex<double>, 1> &d,
    Array<complex<double>, 1> const &a,
    Array<double, 1> const &b,
    Array<complex<double>, 1> const &c
  ) {

    d[0].real() = a[0].real() * b[0] + c[0].real();
    d[0].imag() = a[0].imag() * b[0] + c[0].imag();
  }
};

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<
  gemm::GemmShape<1, 1, 1>,
  1,
  double,
  LayoutA,
  complex<double>,
  LayoutB,
  complex<double>,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAddComplex;
  using ElementC = complex<double>;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<complex<double>, 1> &d,
    Array<double, 1> const &a,
    Array<complex<double>, 1> const &b,
    Array<complex<double>, 1> const &c
  ) {

    d[0].real() = a[0] * b[0].real() + c[0].real();
    d[0].imag() = a[0] * b[0].imag() + d[0].imag();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, half_t, LayoutA, half_t, LayoutB, float, LayoutC, OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAdd;
  using ElementC = float;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<float, 1> &d,
    Array<half_t, 1> const &a,
    Array<half_t, 1> const &b,
    Array<float, 1> const &c
  ) {
    d[0] = float(a[0]) * float(b[0]) + c[0];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation for Quaternions
template <
  /// Layout of A matrix
  typename LayoutA,
  /// Layout of B matrix
  typename LayoutB,
  /// Layout of C matrix
  typename LayoutC
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, Quaternion<float>, LayoutA, Quaternion<float>, LayoutB, Quaternion<float>, LayoutC, OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using Operator = OpMultiplyAdd;
  using Element = Quaternion<float>;
  using ElementC = Element;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<Element, 1> &d,
    Array<Element, 1> const &a,
    Array<Element, 1> const &b,
    Array<Element, 1> const &c
  ) {
    multiply_add<Element, Element, Element> op;
    d[0] = op(a[0], b[0], c[0]);
  }

};

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////
