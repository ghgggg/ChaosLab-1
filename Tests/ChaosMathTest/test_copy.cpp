#include "core.hpp"

namespace chaos
{
	TEST_CLASS(CopyTest)
	{
	public:
		CopyTest() {}

		TEST_METHOD(ContinuaCopyTo)
		{
			float buf[] = { 1,2,3,4,5,6,7,8,9 };
			Tensor A = Tensor(Shape(3,3), Depth::D4, Packing::CHW, buf);
			Tensor B; // empty
			A.CopyTo(B);
			for (int i = 0; i < 9; i++)
			{
				Assert::AreEqual(buf[i], B[i], FLT_EPSILON);
			}

			AutoBuffer<float> c;
			Tensor C = Tensor(Shape(3,3), Depth::D4, Packing::CHW, c.data());
			A.CopyTo(C);
			for (int i = 0; i < 9; i++)
			{
				Assert::AreEqual(buf[i], c[i], FLT_EPSILON);
			}

			AutoBuffer<float> d;
			Tensor D = Tensor(Shape(3, 3), Depth::D4, Packing::CHW, d.data(), {16,1});
			A.CopyTo(D);
			for (int r = 0; r < 3; r++)
			{
				for (int c = 0; c < 3; c++)
				{
					int idx1 = r * 16 + c;
					int idx2 = r * 3 + c;
					Assert::AreEqual(buf[idx2], D[idx1], FLT_EPSILON);
				}
			}
		}

		TEST_METHOD(NonContinuaCopyTo)
		{
			float buf[] = { 1,2,3,0,4,5,6,0,7,8,9,0 };
			Tensor A = Tensor(Shape(3, 3), Depth::D4, Packing::CHW, buf, { 4,1 });
			Tensor B; // empty
			A.CopyTo(B);
			for (int r = 0; r < 3; r++)
			{
				for (int c = 0; c < 3; c++)
				{
					int idx1 = r * 3 + c;
					int idx2 = r * 4 + c;
					Assert::AreEqual(buf[idx2], B[idx1], FLT_EPSILON);
				}
			}

			AutoBuffer<float> c;
			Tensor C = Tensor(Shape(3, 3), Depth::D4, Packing::CHW, c.data(), { 8,1 });
			A.CopyTo(C);
			for (int r = 0; r < 3; r++)
			{
				for (int c = 0; c < 3; c++)
				{
					int idx1 = r * 8 + c;
					int idx2 = r * 4 + c;
					Assert::AreEqual(buf[idx2], C[idx1], FLT_EPSILON);
				}
			}
		}
	};
}