#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		Layer::Layer(const std::string& type) : type(type) 
		{
			one_blob_only = false;
			support_inplace = false;
		}

		void Layer::Forward(const std::vector<Tensor>& bottoms, std::vector<Tensor>& tops, const Option& opt) const
		{
			CHECK(support_inplace) << "not support inplace";

			tops = bottoms;
			for (size_t i = 0; i < tops.size(); i++)
			{
				tops[i] = bottoms[i].Clone(opt.blob_allocator);
				CHECK(not tops[i].empty());
			}

			Forward(tops, opt);
		}
		void Layer::Forward(const Tensor& bottom, Tensor& top, const Option& opt) const
		{
			CHECK(support_inplace) << "not support inplace";

			top = bottom.Clone(opt.blob_allocator);
			CHECK(not top.empty());
			Forward(top, opt);
		}
		void Layer::Forward(std::vector<Tensor>& /*blobs*/, const Option& /*opt*/) const
		{
			LOG(FATAL) << "not implemented";
		}
		void Layer::Forward(Tensor& /*blob*/, const Option& /*opt*/) const
		{
			LOG(FATAL) << "not implemented";
		}

	}
}