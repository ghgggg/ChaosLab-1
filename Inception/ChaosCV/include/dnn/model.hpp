#pragma once

#include "core/tensor.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API ParamValue
		{
		public:
			ParamValue();
			ParamValue(const int& ival);
			ParamValue(const float& fval);
			ParamValue(const std::string& sval);
			ParamValue(const Tensor& tval);

			operator const int& () const;
			operator const float& () const;
			operator const std::string& () const;
			operator const Tensor& () const;

			~ParamValue() = default;
		private:
			enum
			{
				NONE,
				INT,
				FLOAT,
				STRING,
				TENSOR,
			};

			void Init(int type, const void* obj);

			int type;
			void* obj;
		};
	}
}