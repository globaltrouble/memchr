/// A trait for describing vector operations used by vectorized searchers.
///
/// The trait is highly constrained to low level vector operations needed. In
/// general, it was invented mostly to be generic over x86's __m128i and
/// __m256i types. It's likely that once std::simd becomes a thing, we can
/// migrate to that since the operations required are quite simple.
///
/// # Safety
///
/// All methods are not safe since they are intended to be implemented using
/// vendor intrinsics, which are also not safe. Callers must ensure that the
/// appropriate target features are enabled in the calling function, and that
/// the current CPU supports them. All implementations should avoid marking the
/// routines with #[target_feature] and instead mark them as #[inline(always)]
/// to ensure they get appropriately inlined. (inline(always) cannot be used
/// with target_feature.)
pub(crate) trait Vector: Copy + core::fmt::Debug {
    const BITS: usize;
    const BYTES: usize;

    /// Create a vector with 8-bit lanes with the given byte repeated into each
    /// lane.
    unsafe fn splat(byte: u8) -> Self;

    /// Read a vector-size number of bytes from the given pointer. The pointer
    /// must be aligned to the size of the vector.
    ///
    /// # Safety
    ///
    /// Callers must guarantee that at least `BYTES` bytes are readable from
    /// `data` and that `data` is aligned to a `BYTES` boundary.
    unsafe fn load_aligned(data: *const u8) -> Self;

    /// Read a vector-size number of bytes from the given pointer. The pointer
    /// does not need to be aligned.
    ///
    /// # Safety
    ///
    /// Callers must guarantee that at least `BYTES` bytes are readable from
    /// `data`.
    unsafe fn load_unaligned(data: *const u8) -> Self;

    /// _mm_movemask_epi8 or _mm256_movemask_epi8
    unsafe fn movemask(self) -> u32;
    /// _mm_cmpeq_epi8 or _mm256_cmpeq_epi8
    unsafe fn cmpeq(self, vector2: Self) -> Self;
    /// _mm_and_si128 or _mm256_and_si256
    unsafe fn and(self, vector2: Self) -> Self;
    /// _mm_or or _mm256_or_si256
    unsafe fn or(self, vector2: Self) -> Self;
}

#[cfg(target_arch = "x86_64")]
mod x86sse {
    use core::arch::x86_64::*;

    use super::Vector;

    impl Vector for __m128i {
        const BITS: usize = 128;
        const BYTES: usize = 16;

        #[inline(always)]
        unsafe fn splat(byte: u8) -> __m128i {
            _mm_set1_epi8(byte as i8)
        }

        #[inline(always)]
        unsafe fn load_aligned(data: *const u8) -> __m128i {
            _mm_load_si128(data as *const __m128i)
        }

        #[inline(always)]
        unsafe fn load_unaligned(data: *const u8) -> __m128i {
            _mm_loadu_si128(data as *const __m128i)
        }

        #[inline(always)]
        unsafe fn movemask(self) -> u32 {
            _mm_movemask_epi8(self) as u32
        }

        #[inline(always)]
        unsafe fn cmpeq(self, vector2: Self) -> __m128i {
            _mm_cmpeq_epi8(self, vector2)
        }

        #[inline(always)]
        unsafe fn and(self, vector2: Self) -> __m128i {
            _mm_and_si128(self, vector2)
        }

        #[inline(always)]
        unsafe fn or(self, vector2: Self) -> __m128i {
            _mm_or_si128(self, vector2)
        }
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
mod x86avx {
    use core::arch::x86_64::*;

    use super::Vector;

    impl Vector for __m256i {
        const BITS: usize = 256;
        const BYTES: usize = 32;

        #[inline(always)]
        unsafe fn splat(byte: u8) -> __m256i {
            _mm256_set1_epi8(byte as i8)
        }

        #[inline(always)]
        unsafe fn load_aligned(data: *const u8) -> __m256i {
            _mm256_load_si256(data as *const __m256i)
        }

        #[inline(always)]
        unsafe fn load_unaligned(data: *const u8) -> __m256i {
            _mm256_loadu_si256(data as *const __m256i)
        }

        #[inline(always)]
        unsafe fn movemask(self) -> u32 {
            _mm256_movemask_epi8(self) as u32
        }

        #[inline(always)]
        unsafe fn cmpeq(self, vector2: Self) -> __m256i {
            _mm256_cmpeq_epi8(self, vector2)
        }

        #[inline(always)]
        unsafe fn and(self, vector2: Self) -> __m256i {
            _mm256_and_si256(self, vector2)
        }

        #[inline(always)]
        unsafe fn or(self, vector2: Self) -> __m256i {
            _mm256_or_si256(self, vector2)
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm_simd128 {
    use core::arch::wasm32::*;

    use super::Vector;

    impl Vector for v128 {
        const BITS: usize = 128;
        const BYTES: usize = 16;

        #[inline(always)]
        unsafe fn splat(byte: u8) -> v128 {
            u8x16_splat(byte)
        }

        #[inline(always)]
        unsafe fn load_aligned(data: *const u8) -> v128 {
            *data.cast()
        }

        #[inline(always)]
        unsafe fn load_unaligned(data: *const u8) -> v128 {
            v128_load(data.cast())
        }

        #[inline(always)]
        unsafe fn movemask(self) -> u32 {
            u8x16_bitmask(self).into()
        }

        #[inline(always)]
        unsafe fn cmpeq(self, vector2: Self) -> v128 {
            u8x16_eq(self, vector2)
        }

        #[inline(always)]
        unsafe fn and(self, vector2: Self) -> v128 {
            v128_and(self, vector2)
        }

        #[inline(always)]
        unsafe fn or(self, vector2: Self) -> v128 {
            v128_or(self, vector2)
        }
    }
}
