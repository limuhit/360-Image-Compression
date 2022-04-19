import sys
import argparse
from functools import reduce
def copy_one_file(dir,fname, type, prex=''):
    if len(prex)>0:
        fin = open('./template/temp_%s.%s'%(prex,type))
        f = open('%s/%s_%s.%s'%(dir,fname,prex,type),'w')
    else:
        fin = open('./template/temp.%s'%type)
        f = open('%s/%s.%s'%(dir,fname,type),'w')
    for pt in fin.readlines():
        f.write(pt.replace('name_replace',fname))
    fin.close()
    f.close()

def copy_one_file_v2(dir, mdict, type, ver=2, prex=''):
    fname = mdict['name_replace']
    if len(prex)>0:
        fin = open('./template/temp%d_%s.%s'%(ver,prex,type))
        f = open('%s/%s_%s.%s'%(dir,fname,prex,type),'w')
    else:
        fin = open('./template/temp%d.%s'%(ver,type))
        f = open('%s/%s.%s'%(dir,fname,type),'w')
    for pt in fin.readlines():
        tmp = pt
        for pk in mdict.keys():
            tmp = tmp.replace(pk,mdict[pk])
        f.write(tmp)
    fin.close()
    f.close()

def name_for_python(fname):
    tmp = fname.split('_')
    tmp = [pt.capitalize() for pt in tmp]
    return reduce(lambda xa,xb:xa+xb,tmp)

def find_anchor_setup(flist):
    idx = 0
    cuda_flag = False
    while idx < len(flist):
        if flist[idx].find('CUDAExtension(')>=0:
            cuda_flag = True
        if cuda_flag and flist[idx].find(']')>=0:
            return idx
        idx+=1
    return -1
def inject_main(dir, fname, pname, pyinit):
    f = open('%s/main.hpp'%dir,'a')
    f.write('#include "%s.hpp"\n'%fname)
    f.close()
    f = open('%s/main.cpp'%dir,'r')
    contents = f.readlines()
    f.close()
    f = open('%s/main.cpp'%dir,'w')
    ins_list = ['\n','    py::class_<%s_opt>(m,"%sOp")\n'%(fname, pname), '        .def(py::init<%s int, bool>())\n'%pyinit,
                '        .def("to", &%s_opt::to)\n'%fname, '        .def("forward", &%s_opt::forward_cuda)\n'%fname,
                '        .def("backward", &%s_opt::backward_cuda);\n'%fname]
    new_content = contents[:-1]+ins_list+[contents[-1]]
    f.write(''.join(new_content))
    f.close()
    f = open('setup.py','r')
    flist = f.readlines()
    f.close()
    anchor = find_anchor_setup(flist)
    if anchor > 0:
        flist = flist[:anchor] + ["\t\t\t'./extension/%s_cuda.cu',\n"%fname] + flist[anchor:]
        f = open('setup.py', 'w')
        f.write(''.join(flist))
        f.close()

def inject_init_file(dir, pk_name, fname):
    f = open('{}/__init__.py'.format(dir),'a')
    f.write('from {}.{} import {}\n'.format(pk_name, fname, fname))
    f.close()
   
def without_cpp(args):
    ext_dir = './extension'
    python_dir = './lic360_operator'
    ver = 2
    if args.cublas:
        ver = 3
    else:
        if args.tensor: ver = 4
    mdict = produce_var_list(args.p)
    fname = args.name
    mdict['name_replace'] = fname
    copy_one_file_v2(ext_dir, mdict,'cu',ver,'cuda')
    copy_one_file_v2(ext_dir, mdict,'hpp',ver)
    pname = name_for_python(fname)
    mdict['name_replace'] = pname
    copy_one_file_v2(python_dir,mdict,'py',ver)
    inject_init_file(python_dir, python_dir.split('/')[-1], pname)
    inject_main(ext_dir,fname,pname, mdict['python_init,'])
    

def produce_var_list(plist):
    mdict={'var_replace,':'','class_v_replace;':'', 'class_var_init':'', 'python_init,':'', 'var_to_replace,':''}
    try:
        plist = [pt.split('/') for pt in plist]
        mdict['var_replace,'] = ', '.join([pt[1]+' '+pt[0] for pt in plist])+','
        mdict['class_v_replace;'] = ';\n\t\t'.join([pt[1]+' '+ pt[0] + '_' for pt in plist])+';'
        mdict['class_var_init'] = '\n\t\t\t'.join(['%s_ = %s;'%(pt[0],pt[0]) for pt in plist])
        mdict['python_init,'] = ', '.join([pt[1] for pt in plist])+','
        mdict['var_to_replace,'] = ','.join([pt[0] for pt in plist])+',' 
    except:
        pass
    return mdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='New Operator')
    parser.add_argument('-name', '-n')
    parser.add_argument('-cublas', action='store_true', default=False)
    parser.add_argument('-tensor', action='store_true', default=False)
    parser.add_argument('-p', nargs='*', default=[1])
    args = parser.parse_args()
    print(args)
    if args.name is not None:
        without_cpp(args)
        