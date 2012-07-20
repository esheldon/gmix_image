from fabricate import *
import sys, os
import glob
import optparse

parser = optparse.OptionParser()
# make an options list, also send to fabricate
optlist=[optparse.Option('--prefix','-p',
                         default=sys.exec_prefix,help="where to install")]
parser.add_options(optlist)

options,args = parser.parse_args()
prefix=os.path.expanduser( options.prefix )

# set to '' for local directory
sdir='gmix_image/'

CC='gcc'

# -lrt is only needed for the timing stuff
LINKFLAGS=['-lm','-lrt']

CFLAGS=['-std=gnu99','-Wall','-Werror','-O2']

test_sources = ['test-em','gvec','image','gmix_em','matrix','bound']

test_newimage_sources = ['test-newimage','image','bound']

test_cen_sources = ['gvec','image', 'gmix_em','matrix']

test_mat_sources = ['test-matrix','matrix']

test_sources = [sdir+s for s in test_sources]
test_newimage_sources = [sdir+s for s in test_newimage_sources]
test_cen_sources = [sdir+s for s in test_cen_sources]
test_mat_sources = [sdir+s for s in test_mat_sources]

programs = [{'name':sdir+'test-em', 'sources':test_sources},
            {'name':sdir+'test-newimage', 'sources':test_newimage_sources}]


install_targets = [(prog['name'],'bin') for prog in programs]

def build():
    compile()
    link()

def compile():
    for prog in programs:
        for source in prog['sources']:
            run(CC, '-c', '-o',source+'.o', CFLAGS, source+'.c')

def link():
    for prog in programs:
        objects = [s+'.o' for s in prog['sources']]
        run(CC,'-o', prog['name'], objects,LINKFLAGS)

def clean():
    autoclean()

def install():
    import shutil

    # make sure everything is built first
    build()

    for target in install_targets:
        (name,subdir) = target
        subdir = os.path.join(prefix, subdir)
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        dest=os.path.join(subdir, os.path.basename(name))
        sys.stdout.write("install: %s\n" % dest)
        shutil.copy(name, dest)

main(extra_options=optlist)
