#include "cblas.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
//#include <Eigen/Dense>
//#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

namespace tfembed {
    //using Eigen::MatrixXd;

    class Hash {
        std::vector<cv::Mat> ws;
        std::vector<cv::Mat> bs;
        unsigned dim_in;
        unsigned dim_out;
        cv::Mat apply_helper (float *v) {
            cv::Mat m;
            cv::Mat(1, dim_in, CV_32F, v).copyTo(m);
#ifdef TFEMBED_64
            m.convertTo(m, CV_64F);
#endif
            for (unsigned i = 0; i < ws.size(); ++i) {
                cv::Mat tmp;
                /*
                Eigen::Matrix<float, -1, -1> e_m, e_ws, e_bs, e_tmp;
                cv::cv2eigen(m, e_m);
                cv::cv2eigen(ws[i], e_ws);
                cv::cv2eigen(bs[i], e_bs);
                e_tmp = e_m * e_ws + e_bs;
                cv::eigen2cv(e_tmp, tmp);
                */
                cv::gemm(m, ws[i], 1.0, bs[i], 1.0, tmp);

                m = tmp;
                if (i < ws.size() - 1) {
                    float *p = m.ptr<float>(0);
                    for (unsigned j = 0; j < m.cols; ++j) {
                        if (!(p[j] > 0)) p[j] = 0;
                    }
                }
            }
            if (m.cols != dim_out) throw std::runtime_error("size mismatch");
#ifdef TFEMBED_64
            m.convertTo(m, CV_32F);
#endif
            return m;
        }
    public:
        Hash (std::string const &model) {
            std::ifstream is(model.c_str(), std::ios::binary);
            uint32_t nl;
            is.read((char *)&nl, sizeof(nl));
            dim_in = dim_out = 0;
            for (unsigned i = 0; i < nl; ++i) {
                uint32_t din, dout;
                is.read((char *)&din, sizeof(din));
                is.read((char *)&dout, sizeof(dout));
                std::cerr << "loading layer " << din << 'x' << dout << std::endl;
                {
                    cv::Mat mat(din, dout, CV_32F);
                    if (!mat.isContinuous()) throw std::runtime_error("contiguous");
                    is.read((char *)mat.data, din * dout * sizeof(float));
#ifdef TFEMBED_64
                    mat.convertTo(mat, CV_64F);
#endif
                    ws.push_back(mat);
                }
                {
                    cv::Mat mat(1, dout, CV_32F);
                    if (!mat.isContinuous()) throw std::runtime_error("contiguous");
                    is.read((char *)mat.data, dout * sizeof(float));
#ifdef TFEMBED_64
                    mat.convertTo(mat, CV_64F);
#endif
                    bs.push_back(mat);
                }
                if (dim_in == 0) dim_in = din;
                else {
                    if (din != dim_out) throw std::runtime_error("matrix mismatch");
                }
                dim_out = dout;
            }
            if (!is) throw std::runtime_error("error reading model file");
        }

        template <typename T>
        void apply (float *v, T *out) {
            cv::Mat m = apply_helper(v);
            float const *ptr = m.ptr<float>(0);
            if (dim_out % (sizeof(T) * 8) != 0) throw std::runtime_error("bad output size");
            unsigned nc = dim_out / (sizeof(T) * 8);
            for (unsigned i = 0; i < nc; ++i) {
                auto &o = out[i];
                o = 0;
                for (unsigned i = 0; i < (sizeof(T) * 8); ++i) {
                    o = o << 1LL;
                    if (ptr[0] > 0) {
                        o |= 1LL;
                    }
                    ++ptr;
                }
            }
        }

#ifdef TFEMBED_DEBUG
        cv::Mat test () {
            std::vector<float> v;
            float sum = 0;
            for (unsigned i = 0; i < dim_in; ++i) {
                v.push_back(i);
                sum += i;
            }
            for (auto &a: v) a = 0.015625; //a/=sum;
            return apply_helper(&v[0]);
        }
#endif
    };
}

