from distutils.core import setup, Extension

tfembed = Extension('_tfembed',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++1y', '-g'], 
        libraries = ['kgraph', 'boost_timer', 'boost_chrono', 'boost_system', 'boost_python'],
        include_dirs = ['/usr/local/include'],
        library_dirs = ['/usr/local/lib'],
        sources = ['python-api.cpp']
        )

setup (name = 'tfembed',
       version = '0.0.1',
       author = 'Wei Dong',
       author_email = 'wdong@wdong.org',
       license = 'BSD',
       description = 'This is a demo package',
       ext_modules = [tfembed],
       )
