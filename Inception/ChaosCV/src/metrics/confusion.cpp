#include "metrics/confusion.hpp"

namespace chaos
{
	namespace metrics
	{
		/// <summary>
		/// To find the first value with < th, search from start
		/// </summary>
		/// <param name="data">data buffer</param>
		/// <param name="start">start idx</param>
		/// <param name="th">threshold</param>
		/// <returns>the value idx</returns>
		inline int Find1stValue(const Tensor& data, int start, float th)
		{
			const float* _data = data;
			int i = start;
			for (; i < data.shape.vol(); i++)
			{
				if (_data[i] < th) return i;
			}
			return i;
		}

		ConfusionMatrix::ConfusionMatrix()
		{
			threshold.insert(1);
		}

		void ConfusionMatrix::operator()(bool is_pos, float score)
		{
			is_pos ? pos.insert(score) : neg.insert(score);
			threshold.insert(score);
		}

		void ConfusionMatrix::operator()(const Tensor& is_pos, const Tensor& score)
		{
			CHECK_EQ(is_pos.shape.vol(), score.shape.vol()) << "shape missmatch";

			for (int i = 0; i < is_pos.shape.vol(); i++)
			{
				is_pos[i] ? pos.insert(score[i]) : neg.insert(score[i]);
				threshold.insert(score[i]);
			}
		}

		void ConfusionMatrix::Apply()
		{
			int idx;
			idx = 0;
			Tensor _neg = Tensor({ neg.size() }, Depth::D4, Packing::CHW);
			for (const auto& val : neg) { _neg[idx++] = val; }

			idx = 0;
			Tensor _pos = Tensor({ pos.size() }, Depth::D4, Packing::CHW);
			for (const auto& val : pos) { _pos[idx++] = val; }

			table.Create(Shape(4, (int)threshold.size()), Depth::D4, Packing::CHW, nullptr);
			size_t rsteps = table.steps[0];
			int* tp = (int*)table; // .row(0);
			int* fp = (int*)table + rsteps; // .row(1);
			int* fn = (int*)table + 2 * rsteps; // .row(2);
			int* tn = (int*)table + 3 * rsteps; // .row(3);

			idx = 0;
			int ni = 0, pi = 0;
			for (const auto& th : threshold)
			{
				pi = Find1stValue(_pos, pi, th);
				ni = Find1stValue(_neg, ni, th);
				tp[idx] = pi; //std::distance(pos.begin(), pos.upper_bound(th)); //CountGE(_pos, th);
				fp[idx] = ni; // std::distance(neg.begin(), neg.upper_bound(th)); //CountGE(_neg, th);
				fn[idx] = (int)pos.size() - tp[idx];
				tn[idx] = (int)neg.size() - fp[idx];
				idx++;
			}
		}

		Tensor ConfusionMatrix::GetACC() const
		{
			Tensor acc = Tensor({ threshold.size() }, Depth::D4, Packing::CHW);
			size_t rsteps = table.steps[0];
			const int* tp = (const int*)table;
			const int* fp = (const int*)table + rsteps;
			const int* fn = (const int*)table + 2 * rsteps;
			const int* tn = (const int*)table + 3 * rsteps;
			for (int i = 0; i < threshold.size(); i++)
			{
				acc[i] = (tp[i] + tn[i]) / (float)(tp[i] + tn[i] + fp[i] + fn[i]);
			}
			return acc;
		}

		Tensor ConfusionMatrix::GetTPR() const
		{
			Tensor tpr = Tensor({ threshold.size() }, Depth::D4, Packing::CHW);
			//size_t rsteps = table.steps[0];
			const int* tp = (const int*)table;
			for (int i = 0; i < threshold.size(); i++)
			{
				tpr[i] = tp[i] / (float)pos.size();
			}
			return tpr;
		}
		Tensor ConfusionMatrix::GetFPR() const
		{
			Tensor fpr = Tensor({ threshold.size() }, Depth::D4, Packing::CHW);
			size_t rsteps = table.steps[0];
			const int* fp = (const int*)table + rsteps; // .row(1);
			for (int i = 0; i < threshold.size(); i++)
			{
				fpr[i] = fp[i] / (float)neg.size();
			}
			return fpr;
		}
		Tensor ConfusionMatrix::GetPPV() const
		{
			Tensor ppv = Tensor({ threshold.size() }, Depth::D4, Packing::CHW);
			size_t rsteps = table.steps[0];
			const int* tp = (const int*)table;
			const int* fp = (const int*)table + rsteps;
			for (int i = 0; i < threshold.size(); i++)
			{
				float p = (float)tp[i] + fp[i];
				ppv[i] = p < 1e-6 ? 1 : tp[i] / p;
			}
			return ppv;
		}
		Tensor ConfusionMatrix::GetThreshold() const
		{
			Tensor th = Tensor({ threshold.size() }, Depth::D4, Packing::CHW);
			int idx = 0;
			for (const auto& val : threshold)
			{
				th[idx++] = val;
			}
			return th;
		}
	}
}