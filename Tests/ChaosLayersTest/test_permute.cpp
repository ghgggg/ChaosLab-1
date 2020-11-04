#include "core.hpp"

namespace chaos
{
	TEST_CLASS(PermuteTest)
	{
	public:
		PermuteTest()
		{
			A233.Create(Shape(2, 3, 3), {9,3,1}, Depth::D4, Packing::CHW, nullptr);
			for (size_t i = 0; i < A233.shape.vol(); i++)
			{
				A233[i] = (i % 9) + 1;
			}
		}

		TEST_METHOD(P233)
		{
			auto layer = dnn::LayerRegistry::CreateLayer("Permute");
			
		}

		Tensor A233;
	};
}