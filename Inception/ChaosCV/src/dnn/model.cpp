#include "dnn/model.hpp"

namespace chaos
{
	namespace dnn
	{
		ParamValue::ParamValue() { Init(NONE, nullptr); }
		ParamValue::ParamValue(const int& ival) { Init(INT, &ival); }
		ParamValue::ParamValue(const float& fval) { Init(FLOAT, &fval); }
		ParamValue::ParamValue(const std::string& sval) { Init(STRING, &sval); }
		ParamValue::ParamValue(const Tensor& tval) { Init(TENSOR, &tval); }

		ParamValue::operator const int& () const
		{
			CHECK_EQ(type, INT) << "param value is not int";
			return *(int*)obj;
		}
		ParamValue::operator const float& () const
		{
			CHECK_EQ(type, FLOAT) << "param value is not float";
			return *(float*)obj;
		}
		ParamValue::operator const std::string& () const
		{
			CHECK_EQ(type, STRING) << "param value is not string";
			return *(std::string*)obj;
		}
		ParamValue::operator const Tensor& () const
		{
			CHECK_EQ(type, TENSOR) << "param value is not tensor";
			return *(Tensor*)obj;
		}

		void ParamValue::Init(int _type, const void* _obj) { type = _type; obj = (void*)_obj; }
	}
}