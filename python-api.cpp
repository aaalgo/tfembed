#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/python.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/raw_function.hpp>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <kgraph.h>
#include <kgraph-data.h>
#include <numpy/ndarrayobject.h>
#define TFEMBED_DEBUG 1
//#define TFEMBED_64 1
#include "tfembed.h"
using namespace boost::python;

namespace kgraph {
    namespace metric {
        /// L2 square distance.
        struct chi2 {
            template <typename T>
            /// L2 square distance.
            static float apply (T const *t1, T const *t2, unsigned dim) {
                float r = 0;
                for (unsigned i = 0; i < dim; ++i) {
                    float v = float(t1[i]) - float(t2[i]);
                    float b = float(t1[i]) + float(t2[i]);
                    if (b > 0) {
                        r += v * v / b;
                    }
                }
                return r;
            }
        };
    }
}

namespace {
    using std::ofstream;
    using std::ostream;
    using std::ios;
    using std::string;
    using std::vector;
    using std::runtime_error;
    using std::cerr;
    using std::endl;
    using std::fill;
    using kgraph::Matrix;
    using kgraph::MatrixOracle;
    using kgraph::KGraph;

    class SampleStream {
        Matrix<float> data;
        KGraph *knns;
        std::default_random_engine rng;
        vector<unsigned> index;
        unsigned off;
        vector<unsigned> nns_buf;
        void copy (unsigned i, float *buf) {
            float const *from = data[i];
            std::copy(from, from + data.dim(), buf);
        }
    public:
        SampleStream (string db, string cache): rng(2017) {
            data.load_lshkit(db);
            cerr << "Loaded data " << data.size() << 'x' << data.dim() << endl;
            //C.P = extract<decltype(C.P)>(kwargs.get(#P, C.P)) 
            MatrixOracle<float, kgraph::metric::chi2> chi2oracle(data);
            KGraph::IndexParams params;
            knns = KGraph::create();
            if (boost::filesystem::exists(cache)) {
                knns->load(cache.c_str());
            }
            else {
                knns->build(chi2oracle, params, NULL);
                knns->save(cache.c_str());
            }
            index.resize(data.size());
            for (unsigned i = 0; i < index.size(); ++i) index[i] = i;
            off = index.size();
            if (index.size() < params.L) throw runtime_error("dataset too small");
            if (rng.max() < data.size()) throw runtime_error("rng range too small");
            nns_buf.resize(params.L);
        }
        ~SampleStream () {
            delete knns;
        }
        object next () {
            npy_intp dims[] = {3, data.dim()};
            PyArrayObject *triplet = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT);
            auto stride = triplet->strides[0];
            float *p1 = (float *)(triplet->data);
            float *p2 = (float *)(triplet->data + stride);
            float *p3 = (float *)(triplet->data + stride * 2);

            if (off >= index.size()) {
                std::shuffle(index.begin(), index.end(), rng);
                off = 0;
            }
            unsigned ref = index[off++];
            unsigned near, far;
            unsigned M, L;
            knns->get_nn(ref, &nns_buf[0], &M, &L);
            near = rng() % L;
            far = rng() % L;
            if (near < far) {
                near = nns_buf[near];
                far = nns_buf[far];
            }
            else {
                near = nns_buf[near];
                for (;;) {
                    far = rng() % data.size();
                    if (far != near && far != ref) break;
                }
                float d_near = kgraph::metric::chi2::apply<float>(data[ref], data[near], data.dim());
                float d_far = kgraph::metric::chi2::apply<float>(data[ref], data[far], data.dim());
                if (d_far < d_near) std::swap(near, far);
            }
            copy(ref, p1);
            copy(near, p2);
            copy(far, p3);
            return object(boost::python::handle<>((PyObject *)triplet));
        }
        int dim () const {
            return data.dim();
        }
    };

    object return_iterator (tuple args, dict kwargs) {
        object self = args[0];
        return self;
    };

    struct FlipChoice {
        float margin;
        unsigned vec;
        unsigned dim;
    };

    bool operator < (FlipChoice const &c1, FlipChoice const &c2) {
        return c1.margin < c2.margin;
    }

    template <typename T> int sign(T val) {
        return (T(0) < val) - (val < T(0));
    }

    tuple eval_mask (PyObject *_array) {
        PyArrayObject *array = (PyArrayObject *)_array;
        if (array->nd != 2) throw runtime_error("xxx");
        npy_intp dims[] = {array->dimensions[0],
                           array->dimensions[1]};
        unsigned D = dims[1];
        // assert array->nd == 2
        auto stride = array->strides[0];
        float *p1 = (float *)(array->data);
        float *p2 = (float *)(array->data + stride);
        float *p3 = (float *)(array->data + stride * 2);

        /*
        PyArrayObject *mask = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT);
        stride = mask->strides[0];
        if (stride != D * sizeof(float)) throw runtime_error("bad size");
        float *m1 = (float *)(mask->data);
        float *m2 = (float *)(mask->data + stride);
        float *m3 = (float *)(mask->data + stride * 2);
        fill(m1, m1 + D, 0);
        fill(m2, m2 + D, 0);
        fill(m3, m3 + D, 0);

        PyArrayObject *Mask = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT);
        stride = Mask->strides[0];
        float *M1 = (float *)(Mask->data);
        float *M2 = (float *)(Mask->data + stride);
        float *M3 = (float *)(Mask->data + stride * 2);
        fill(M1, M1 + D, 0);
        fill(M2, M2 + D, 0);
        fill(M3, M3 + D, 0);
        */

        int d0 = 0, d1 = 0, d2 = D;
        //vector<FlipChoice> ccs;
        for (unsigned i = 0; i < D; ++i) {
            //FlipChoice cc;
            //cc.dim = i;
            if (p1[i] > 0) ++d0;
            if (p1[i] * p2[i] < 0) {
                ++d1;
                /*
                cc.margin = std::abs(p2[i]);
                cc.vec = 0;
                */
            }
            if (p1[i] * p3[i] > 0) {
                --d2;
                /*
                cc.margin = std::abs(p3[i]);
                cc.vec = 1;
                */
            }
            /*
            ccs.push_back(cc);
            if (abs(p1[i]) < 0.1) M1[i] = sign(p1[i]);
            if (abs(p2[i]) < 0.1) M2[i] = sign(p2[i]);
            if (abs(p3[i]) < 0.1) M3[i] = sign(p3[i]);
            */
        }
        float rr = 1.0 * d0 / D;

        int ok = 1;
        if (d1 >= d2) ok = 0;
        /*
        if (d1 > d2 - margin) {
            std::sort(ccs.begin(), ccs.end());
            unsigned gap = d1 - d2 + margin;
            if (ccs.size() > gap) ccs.resize(gap);
        }
        else {
            ccs.clear();
        }
        for (auto const &cc: ccs) {
            if (cc.vec == 0) {
                if (p2[cc.dim] > 0) {
                    m2[cc.dim] = 1.0;
                }
                else {
                    m2[cc.dim] = -1.0;
                }

            }
            else {
                if (p3[cc.dim] > 0) {
                    m3[cc.dim] = 1.0;
                }
                else {
                    m3[cc.dim] = -1.0;
                }
            }
        }
        */
        return make_tuple(ok, rr);
            /*
            , int(ccs.size()),
                object(boost::python::handle<>((PyObject *)mask)),
                object(boost::python::handle<>((PyObject *)Mask)));
                */
    }

    void save_array (ostream &os, object _w, object _b) {
        PyArrayObject *w = (PyArrayObject *)_w.ptr();
        PyArrayObject *b = (PyArrayObject *)_b.ptr();
        if (w->nd != 2) throw runtime_error("0");
        if (b->nd != 1) throw runtime_error("1");
        if (w->dimensions[1] != b->dimensions[0]) throw runtime_error("2");
        uint32_t din = w->dimensions[0];
        uint32_t dout = w->dimensions[1];
        cerr << "saving layer " << din << "x" << dout << endl;
        os.write((char const *)&din, sizeof(din));
        os.write((char const *)&dout, sizeof(dout));
        if (w->strides[0] != dout * sizeof(float)) throw runtime_error("3");
        if (w->strides[1] != sizeof(float)) throw runtime_error("4");
        if (b->strides[0] != sizeof(float)) throw runtime_error("5");
        os.write(w->data, din * dout * sizeof(float));
        os.write(b->data, dout * sizeof(float));
    }

    object save_model (string path, list params) {
        {
            ofstream os(path.c_str(), ios::binary);
            uint32_t layers = len(params)/2;
            os.write((char const *)&layers, sizeof(layers));
            for (unsigned i = 0; i < layers; ++i) {
                save_array(os, extract<object>(params[2*i]), extract<object>(params[2*i+1]));
            }
            os.flush();
        }
        tfembed::Hash hash(path);
        cv::Mat m = hash.test();
        npy_intp dims[] = {1, m.cols};
        PyArrayObject *arr = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT);
        std::copy(m.ptr<float>(0),
                  m.ptr<float>(0) + m.cols,
                  (float *)arr->data);
        return object(boost::python::handle<>((PyObject *)arr));
    }
}

BOOST_PYTHON_MODULE(_tfembed)
{
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");
    scope().attr("__doc__") = "tfembed C++ code";
    class_<SampleStream, boost::noncopyable>("SampleStream", init<string, string>())
        .def("__iter__", raw_function(return_iterator))
        .def("next", &SampleStream::next)
        .def("dim", &SampleStream::dim)
    ;
    def("eval_mask", ::eval_mask);
    def("save_model", ::save_model);
}

