#ifndef RISA_TIFF_L_H_
#define RISA_TIFF_L_H_

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <tiffio.h>

#include <boost/log/trivial.hpp>

#include <ddrf/Image.h>
#include <ddrf/MemoryPool.h>

namespace risa
{
	namespace loaders
	{
		namespace detail
		{
			template<class T, bool = std::is_integral<T>::value, bool = std::is_unsigned<T>::value> struct SampleFormat {};
			template<class T> struct SampleFormat<T, true, true> { static constexpr auto value = SAMPLEFORMAT_UINT; };
			template<class T> struct SampleFormat<T, true, false> { static constexpr auto value = SAMPLEFORMAT_INT; };
			template<> struct SampleFormat<float> { static constexpr auto value = SAMPLEFORMAT_IEEEFP; };
			template<> struct SampleFormat<double>{ static constexpr auto value = SAMPLEFORMAT_IEEEFP; };

			template<class T> struct BitsPerSample { static constexpr auto value = sizeof(T) << 3; };

			struct TIFFDeleter { auto operator()(TIFF* p) -> void { TIFFClose(p); }};
		}

		class TIFF : public MemoryManager
		{
			public:
            using value_type = float;
				using manager_type = ddrf::cuda::HostMemoryManager<value_type, ddrf::cuda::async_copy_policy>;

			public:
				TIFF(const std::string& address, const std::string& configFile);

				auto loadImage() -> Image<MemoryManager>;

			protected:
				~TIFF();


			private:
            auto readConfig(const std::string& configFile) -> bool;

				unsigned int memoryPoolIndex_;
            std::string inputPath_;
            int numberOfDetectors_;
            int numberOfProjections_;
		};
	}
}


#endif /* DDRF_TIFF_L_ */
