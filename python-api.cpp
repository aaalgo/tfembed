#include <string>
#include <fstream>
#include <iostream>
#include <boost/ref.hpp>
#include <boost/python.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/raw_function.hpp>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <kgraph.h>
#include <kgraph-data.h>
#include <numpy/ndarrayobject.h>
using namespace boost::python;

namespace {
    using std::string;
    using std::runtime_error;
    using std::cerr;
    using std::endl;
    using kgraph::Matrix;

    class SampleStream {
        Matrix<float> data;

    public:
        SampleStream (string db) {
            data.load_lshkit(db);
            cerr << data.size() << 'x' << data.dim() << endl;
            //C.P = extract<decltype(C.P)>(kwargs.get(#P, C.P)) 
        }
        object next () {
            npy_intp dims[] = {3, data.dim()};
            object triplet = object(boost::python::handle<>(PyArray_SimpleNew(2, dims, NPY_FLOAT)));
            return triplet;
        }
        int dim () const {
            return data.dim();
        }
    };

    object return_iterator (tuple args, dict kwargs) {
        object self = args[0];
        return self;
    };

    tuple eval_mask (PyObject *_array) {
        PyArrayObject *array = (PyArrayObject *)_array;
        if (array->nd != 2) throw runtime_error("xxx");
        npy_intp dims[] = {array->dimensions[0],
                           array->dimensions[1]};
        // assert array->nd == 2
        object mask = object(boost::python::handle<>(PyArray_SimpleNew(2, dims, NPY_FLOAT)));
        // evaluate
        return make_tuple(mask,2,3);
    }
}

BOOST_PYTHON_MODULE(_tfembed)
{
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");
    scope().attr("__doc__") = "tfembed C++ code";
    class_<SampleStream, boost::noncopyable>("SampleStream", init<string>())
        .def("__iter__", raw_function(return_iterator))
        .def("next", &SampleStream::next)
        .def("dim", &SampleStream::dim)
    ;
    def("eval_mask", ::eval_mask);
}

