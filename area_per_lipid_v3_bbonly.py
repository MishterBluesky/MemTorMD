import sys,os
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel

jax.config.update("jax_debug_nans",True) 
#input pdb file, output file and LIPID:BEAD for each lipid where lipid is the lipid name and bead is a representative bead for the lipid, eg POPE:P04
#eg python area_per_lipid_v3.py input.pdb output.pdb POPE:PO4 POPG:PO4

code_type = sys.argv[1]
file_name = sys.argv[2]
lipids = sys.argv[4:]
outfn = sys.argv[3]


def write_point(points,restypes,bvals,fn,add):
	new_file = open(fn+add,"w")
	count = 0
	for n,i in enumerate(points):
		count += 1
		count_str = (6-len(str(count)))*" "+str(count)
		c = "ATOM "+count_str+" BB   "+restypes[n]+"    1       0.000   0.000  15.000  1.00  0.00" 
		xp = np.format_float_positional(i[0],precision=3)
		yp = np.format_float_positional(i[1],precision=3)
		zp = np.format_float_positional(i[2],precision=3)
		bbp = np.format_float_positional(bvals[n],precision=3)
		xp += "0"*(3-len((xp.split(".")[1])))
		yp += "0"*(3-len((yp.split(".")[1])))
		zp += "0"*(3-len((zp.split(".")[1])))
		bbp += "0"*(3-len((bbp.split(".")[1])))
		new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zp))) +zp+c[54:60]+(" "*(8-len(bbp)))+bbp+c[66:]+"\n"
		new_file.write(new_c)
	new_file.close()

def put_into_grid3d(x,r,n,s):
    cx = x[0]-s[0]
    cy = x[1]-s[1]
    cz = x[2]-s[2]
    norm_x = cx/r[0]
    norm_y = cy/r[1]
    norm_z = cz/r[2]
    grid_x = n[0]*norm_x
    grid_y = n[1]*norm_y
    grid_z = n[2]*norm_z
    return int(np.floor(grid_x)),int(np.floor(grid_y)),int(np.floor(grid_z))
    
def get_box_slice(cut_points,points,p,r):
    cpslice = cut_points[points[:,0]>p[0]-r]
    pslice = points[points[:,0]>p[0]-r]
    cpslice = cpslice[pslice[:,0]<p[0]+r]
    pslice = pslice[pslice[:,0]<p[0]+r]
    cpslice = cpslice[pslice[:,1]>p[1]-r]
    pslice = pslice[pslice[:,1]>p[1]-r]
    cpslice = cpslice[pslice[:,1]<p[1]+r]
    pslice = pslice[pslice[:,1]<p[1]+r]
    cpslice = cpslice[pslice[:,2]>p[2]-r]
    pslice = pslice[pslice[:,2]>p[2]-r]
    cpslice = cpslice[pslice[:,2]<p[2]+r]
    pslice = pslice[pslice[:,2]<p[2]+r]
    return pslice,cpslice
    
def center_poses(p,poses,dims):
    cposes = poses - p + dims/2
    cposes[cposes[:,0]<0] += np.array([dims[0],0,0])
    cposes[cposes[:,0]>dims[0]] -= np.array([dims[0],0,0])
    cposes[cposes[:,1]<0] += np.array([0,dims[1],0])
    cposes[cposes[:,1]>dims[1]] -= np.array([0,dims[1],0])
    cposes[cposes[:,2]<0] += np.array([0,0,dims[2]])
    cposes[cposes[:,2]>dims[2]] -= np.array([0,0,dims[2]])
    return cposes

@jax.jit
def center_poses_jax(p,poses,dims):
    cposes = poses - p + dims/2 + dims
    cposes = jnp.fmod(cposes,dims)
    return cposes


    
def allocate_to_grid(poses,r,n,s):
    grid_poses = np.zeros((n[0],n[1],n[2],poses.shape[0]+1),dtype=int)
    for i,p in enumerate(poses):
        indx,indy,indz = put_into_grid3d(p,r,n,s)
        grid_poses[indx,indy,indz,int(grid_poses[indx,indy,indz,-1])] = i
        grid_poses[indx,indy,indz,-1] += 1
    max_poses = np.max(grid_poses[:,:,:,-1])
    grid_poses_cut = grid_poses[:,:,:,:int(max_poses)]
    return grid_poses_cut,grid_poses[:,:,:,-1]

def get_rot_mat(v1,v2):
    v = np.cross(v1,v2)
    angl = np.dot(v1,v2)
    if(angl < -1+1e-6):
        rot_mat = -np.eye(3)
    else:
        vmat = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        rot_mat = np.eye(3)+vmat+np.dot(vmat,vmat)*1/((1+angl))
    return rot_mat
    
                    
@jax.jit      
def add_triangle(tris,new_t,ind):
    direc1 = new_t[1]-new_t[0]
    direc2 = new_t[2]-new_t[0]
    direc1 = jnp.array([direc1[0],direc1[1],0])
    direc2 = jnp.array([direc2[0],direc2[1],0])
    
    view = jnp.cross(direc1,direc2)
    
    add_tri = jnp.zeros((3,2))

    def nadd_flip(add_tri):
        add_tri = add_tri.at[0].set(new_t[0])
        add_tri = add_tri.at[1].set(new_t[1])
        add_tri = add_tri.at[2].set(new_t[2])
        return add_tri
    def add_flip(add_tri):
        add_tri = add_tri.at[0].set(new_t[0])
        add_tri = add_tri.at[1].set(new_t[2])
        add_tri = add_tri.at[2].set(new_t[1])
        return add_tri
    add_tri = jax.lax.cond(jnp.dot(view,jnp.array([0,0,1]))>0,nadd_flip,add_flip,add_tri)
    tris = tris.at[ind].set(add_tri)
    return tris
   
@jax.jit
def in_circumcirc_jax(tria,d):
    row_a = [tria[0][0],tria[0][1],tria[0][0]*tria[0][0]+tria[0][1]*tria[0][1],1]
    row_b = [tria[1][0],tria[1][1],tria[1][0]*tria[1][0]+tria[1][1]*tria[1][1],1]
    row_c = [tria[2][0],tria[2][1],tria[2][0]*tria[2][0]+tria[2][1]*tria[2][1],1]
    row_d = [d[0],d[1],d[0]*d[0]+d[1]*d[1],1]
    mat = jnp.array([row_a,row_b,row_c,row_d])
    return jnp.linalg.det(mat)>0
 
@jax.jit      
def get_cen_jax(tria):
    a = tria[0]
    b = tria[1]
    c = tria[2]
    a2 = a[0]*a[0]+a[1]*a[1]
    b2 = b[0]*b[0]+b[1]*b[1]
    c2 = c[0]*c[0]+c[1]*c[1]
    mat11 = jnp.array([[a[0],a[1],1],[b[0],b[1],1],[c[0],c[1],1]])
    mat12 = jnp.array([[a2,a[1],1],[b2,b[1],1],[c2,c[1],1]])
    mat13 = jnp.array([[a2,a[0],1],[b2,b[0],1],[c2,c[0],1]])
    mat11_det = jnp.linalg.det(mat11)
    cenx = 0.5*jnp.linalg.det(mat12)/mat11_det
    ceny = -0.5*jnp.linalg.det(mat13)/mat11_det
    return jnp.array([cenx,ceny])

@partial(jax.jit,static_argnums=(2))
def delaunay_cen_step_jax(triangles,new_p,totp):
    bad_triangles = jnp.zeros((totp+1,3,2))
    good_triangles = jnp.zeros((totp+1,3,2))
    all_triangles = jnp.concatenate((bad_triangles,good_triangles))
 
            
    def goodbad_loop(all_triangles,ind):
        def go(all_triangles):
            def good(all_triangles):
                inder = jnp.array(all_triangles[totp*2+1][0][0],dtype=int)+totp+1
                all_triangles = all_triangles.at[inder].set(triangles[ind])
                all_triangles = all_triangles.at[totp*2+1].set(all_triangles[totp*2+1]+1)
                return all_triangles
            def bad(all_triangles):
                inder = jnp.array(all_triangles[totp][0][0],dtype=int)
                all_triangles = all_triangles.at[inder].set(triangles[ind])
                all_triangles = all_triangles.at[totp].set(all_triangles[totp]+1)
                return all_triangles
            all_triangles = jax.lax.cond(in_circumcirc_jax(triangles[ind],new_p),bad,good,all_triangles)
            return all_triangles
        def nogo(all_triangles):
            return all_triangles
        all_triangles = jax.lax.cond(jnp.linalg.norm(triangles[ind])>1e-5,go,nogo,all_triangles)
        return all_triangles,ind
    all_triangles,_ = jax.lax.scan(goodbad_loop,all_triangles,jnp.arange(totp))
    
    good_triangles = all_triangles[totp+1:]
    bad_triangles = all_triangles[:totp+1]
    
    def nempty(good_triangles):
        verts = bad_triangles[:-1].reshape(totp*3,2)
        verts = jnp.unique(verts,axis=0,size=totp,fill_value=verts[0]) 
        angles = jnp.array([jnp.arctan2(n[1]-new_p[1],n[0]-new_p[0]) for n in verts])
        order = jnp.argsort(angles) 
        verts = verts[order] 
        verts = jnp.concatenate([verts,jnp.array([verts[0]])]) 
        def add_tris(good_triangles,ind):
            def adder(good_triangles):
                inder = jnp.array(good_triangles[totp],dtype=int)
                good_triangles = add_triangle(good_triangles,jnp.array([verts[ind],verts[ind+1],new_p]),inder)
                good_triangles = good_triangles.at[totp].set(good_triangles[totp]+1)
                return good_triangles
            def nadder(good_triangles):
                return good_triangles 
            good_triangles = jax.lax.cond(((jnp.linalg.norm(verts[ind]) < 1e-5)+(jnp.linalg.norm(verts[ind+1]) < 1e-5)+(jnp.linalg.norm(new_p) < 1e-5))*(jnp.linalg.norm(verts[ind]-verts[ind+1])>1e-5),adder,nadder,good_triangles)
            return good_triangles,ind
        good_triangles,_ = jax.lax.scan(add_tris,good_triangles,jnp.arange(verts.shape[0]))
        return good_triangles
    def empty(good_triangles):
        return good_triangles
    good_triangles = jax.lax.cond(bad_triangles[-1][0][0] > 0,nempty,empty,good_triangles)
    return good_triangles

@jax.jit
def get_delaunay_cen_jax(triangles,points):
    def triscan(triangles,ind):
        triangles = delaunay_cen_step_jax(triangles,points[ind],points.shape[0])
        return triangles,ind
    triangles,_ =jax.lax.scan(triscan,triangles,jnp.arange(points.shape[0]))
    return triangles
    
@jax.jit
def calc_voronoi(triangles):
    cens = jnp.zeros((triangles.shape[0],2))
    def get_all_cens(cens,ind):
        cens = cens.at[ind].set(get_cen_jax(triangles[ind]))
        return cens,ind
    cens,_ = jax.lax.scan(get_all_cens,cens,jnp.arange(triangles.shape[0]))
    angles = jnp.zeros(triangles.shape[0])
    def ang_calc(angles,ind):
        angles = angles.at[ind].set(jnp.arctan2(cens[ind][1],cens[ind][0]))
        return angles,ind
    angles,_ = jax.lax.scan(ang_calc,angles,jnp.arange(triangles.shape[0]))
    order = jnp.argsort(angles) 
    cens = cens[order] 
    cens = jnp.concatenate([cens,jnp.array([cens[0]])])
    tot_area = 0
    def calc_area(tot_area,ind):
        tot_area +=  (cens[ind][0]*(cens[ind+1][1]) + cens[ind+1][0]*(-cens[ind][1]))/2
        return tot_area,ind
    tot_area,_ = jax.lax.scan(calc_area,tot_area,jnp.arange(triangles.shape[0]))
    return tot_area





@jax.jit
def get_normals(poses,tail_dirs):
    norms = jnp.zeros_like(poses)
    def norloop(norms,ind):
        def nor2loop(total,ind2):
            acc_total = total[0]
            number = total[1]
            def close(acc_total,number):
                acc_total += tail_dirs[ind2]
                number +=1
                return acc_total,number
            def far(acc_total,number):
                return acc_total,number
            acc_total,number = jax.lax.cond((jnp.linalg.norm(poses[ind]-poses[ind2]) < 100)*(jnp.dot(tail_dirs[ind],tail_dirs[ind2]) >0.3),close,far,acc_total,number)
            return (acc_total,number),ind2
        total,_ = jax.lax.scan(nor2loop,(jnp.array([0.0,0.0,0.0]),0.0),jnp.arange(tail_dirs.shape[0]))
        direc = total[0]/total[1]
        #direc = poses[ind]-av_pos
        direc /= jnp.linalg.norm(direc)
        norms = norms.at[ind].set(direc)
        return norms,ind
    norms,_ = jax.lax.scan(norloop,norms,jnp.arange(poses.shape[0]))
    return norms

@jax.jit
def get_normalsv2(poses,tail_dirs):
    norms = jnp.zeros_like(poses)
    def norloop(norms,ind):
        test_pos = poses[ind]+tail_dirs[ind]*5
        def nor2loop(total,ind2):
            acc_total = total[0]
            number = total[1]
            def close(acc_total,number):
                hdirec = poses[ind2]-test_pos
                hdirec /= jnp.linalg.norm(hdirec)
                acc_total += hdirec
                number +=1
                return acc_total,number
            def far(acc_total,number):
                return acc_total,number
            acc_total,number = jax.lax.cond(jnp.linalg.norm(poses[ind]-poses[ind2]) < 100,close,far,acc_total,number)
            return (acc_total,number),ind2
        total,_ = jax.lax.scan(nor2loop,(jnp.array([0.0,0.0,0.0]),0.0),jnp.arange(tail_dirs.shape[0]))
        direc = -total[0]/total[1]
        #direc = poses[ind]-av_pos
        direc /= jnp.linalg.norm(direc)
        norms = norms.at[ind].set(direc)
        return norms,ind
    norms,_ = jax.lax.scan(norloop,norms,jnp.arange(poses.shape[0]))
    return norms


def norm_loss(pos,poses,theta,phi):
    direc = jnp.array([jnp.sin(theta)*jnp.sin(phi),jnp.sin(theta)*jnp.cos(phi),jnp.cos(theta)])
    def nlloop(pot,ind):
        def not_zero(pot):
            direc_rep = pos-poses[ind]
            dr_norm = jnp.linalg.norm(direc_rep)
            dr_unit = direc_rep/dr_norm
            ang = jnp.arccos(jnp.dot(dr_unit,direc))
            pot += 0.001/(jnp.power(ang,2)+1e-6) #0.01/(jnp.power(dr_norm,1)+1e-6))*
            return pot
        def zero(pot):
            return pot
        pot = jax.lax.cond(jnp.linalg.norm(pos-poses[ind]) > 1e-5,not_zero,zero,pot)
        return pot,ind
    pot,_ = jax.lax.scan(nlloop,0.0,jnp.arange(poses.shape[0]))
    #jax.debug.print("{x},{y},{w}",x=direc,y=pot,w=pos)
    return pot

norm_grad = jax.grad(norm_loss,argnums=(2,3))

@jax.jit
def minimise_norm(pos,poses,start_dir):
    start_theta = jnp.arccos(jnp.clip(start_dir[2],-1,1))
    start_phi = jnp.arccos(jnp.clip(start_dir[1]/jnp.sin(start_theta),-1,1))
    #jax.debug.print("{x},{y},{z},{w}",x=start_theta,y=start_phi,z=jnp.linalg.norm(start_dir),w=start_dir[1]/jnp.sin(start_theta))
    dt = 0.8
    iters = 500
    def min_loop(angs,ind):
        theta =angs[0]
        phi = angs[1]
        tchange,pchange=norm_grad(pos,poses,theta,phi)
        theta += -tchange*dt
        phi += -pchange*dt
        theta = jnp.fmod(theta,2*jnp.pi)
        phi = jnp.fmod(phi,2*jnp.pi)
        return (theta,phi),ind
    angs,_ = jax.lax.scan(min_loop,(start_theta,start_phi),jnp.arange(iters))
    theta = angs[0]
    phi = angs[1]
    tchange,pchange=norm_grad(pos,poses,theta,phi)
    #jax.debug.print("{x},{y}",x=tchange,y=pchange)
    return jnp.array([jnp.sin(theta)*jnp.sin(phi),jnp.sin(theta)*jnp.cos(phi),jnp.cos(theta)])

@jax.jit
def get_normalsv3(poses,tail_dirs):
    norms = jnp.zeros_like(poses)
    def norloop(norms,ind):
        cposes = center_poses_jax(poses[ind],poses,dims)
        test_poses = jnp.zeros((50,3))+cposes[ind]
        def nor2loop(total,ind2):
            test_poses = total[0]
            cind = jnp.array(total[1],dtype=int)
            def close(test_poses,cind):
                test_poses = test_poses.at[cind].set(cposes[ind2])
                cind += 1
                return test_poses,cind
            def far(test_poses,cind):
                return test_poses,cind
            test_poses,cind = jax.lax.cond(jnp.linalg.norm(cposes[ind]-cposes[ind2]) <50,close,far,test_poses,cind)
            return (test_poses,cind),ind2
        total,_ = jax.lax.scan(nor2loop,(test_poses,0),jnp.arange(cposes.shape[0]))
        #jax.debug.print("{x}",x=total[1])
        test_poses = total[0]
        direc = minimise_norm(cposes[ind],test_poses,tail_dirs[ind])
        norms = norms.at[ind].set(direc*jnp.sign(jnp.dot(direc,tail_dirs[ind])))
        return norms,ind
    norms,_ = jax.lax.scan(norloop,norms,jnp.arange(poses.shape[0]))
    return norms

@partial(jax.jit,static_argnums=2)
def smooth_normals(poses,normals,cutoff):
    new_normals = jnp.zeros_like(normals)
    def norloop(new_normals,ind):
        cposes = center_poses_jax(poses[ind],poses,dims)
        def nor2loop(total,ind2):
            acc_total = total[0]
            number = total[1]
            def close(acc_total,number):
                acc_total += normals[ind2]*(1.0/(jnp.linalg.norm(cposes[ind]-cposes[ind2])+1e-5))
                number +=(1.0/(jnp.linalg.norm(cposes[ind]-cposes[ind2])+1e-5))
                return acc_total,number
            def far(acc_total,number):
                return acc_total,number
            acc_total,number = jax.lax.cond(jnp.linalg.norm(cposes[ind]-cposes[ind2]) < cutoff,close,far,acc_total,number)
            return (acc_total,number),ind2
        total,_ = jax.lax.scan(nor2loop,(jnp.array([0.0,0.0,0.0]),0.0),jnp.arange(normals.shape[0]))
        direc = total[0]/total[1]
        #direc = poses[ind]-av_pos
        direc /= jnp.linalg.norm(direc)
        new_normals = new_normals.at[ind].set(direc)
        return new_normals,ind
    new_normals,_ = jax.lax.scan(norloop,new_normals,jnp.arange(poses.shape[0]))
    return new_normals


def flip_wrong_normals(poses,normals):
    for i in range(3):
        for ind in range(normals.shape[0]):
            cposes = center_poses(poses[ind],poses,dims)
            pgrid,grid_normals = get_box_slice(normals,cposes,dims/2,50)
            order = np.argsort(np.linalg.norm(pgrid-cposes[ind],axis=1))
            grid_normals = grid_normals[order]
            grid_normals = grid_normals[1:6]
            #print(np.dot(grid_normals,normals[ind]))
            angs = np.sign(np.sign(np.sum(np.sign(np.dot(grid_normals,normals[ind])-0.2)))*2+1)
            if(angs < 0):
                normals[ind] = grid_normals[0]
    return normals





    

@partial(jax.jit,static_argnums=(2,3))
def smooth_loop(poses,normals,n,coff):
    def smloop(normals,ind):
        normals = smooth_normals(poses,normals,coff)
        return normals,ind
    normals,_=jax.lax.scan(smloop,normals,jnp.arange(n))
    return normals

@jax.jit
def calc_curvature(pgrid,ngrid):
    cen_pos = jnp.zeros(3)
    cen_normal = jnp.array([0,0,1])

    curves = jnp.zeros(ngrid.shape[0])
    direcs = jnp.zeros(ngrid.shape[0])

    def nloop(data,ind):
        def nzero(data):

            tnorm = ngrid[ind]

            direc2 = pgrid[ind][:2]
            direc2 /= jnp.linalg.norm(direc2)
            direc3 = jnp.concatenate((direc2,jnp.array([0])))
            lend = jnp.dot(tnorm,direc3)
            new_tnorm = lend*direc3+jnp.array([0,0,tnorm[2]])

            new_tnorm /= jnp.linalg.norm(new_tnorm)
            
            

            dzcross = jnp.sign(pgrid[ind][0]/new_tnorm[0])*jnp.linalg.norm(pgrid[ind][:2])/jnp.linalg.norm(new_tnorm[:2])

            
            zcross = -dzcross*new_tnorm[2]+pgrid[ind][2]



            r1 = zcross
            r2 = jnp.linalg.norm(pgrid[ind]-jnp.array([0,0,zcross]))


            sing1 = jnp.sign(zcross)
            sing2 = -jnp.sign(jnp.dot(new_tnorm,pgrid[ind]-jnp.array([0,0,zcross])))
            #jax.debug.print("HERE{x},{y}",x=jnp.dot(new_tnorm,pgrid[ind]-jnp.array([0,0,zcross])),y=jnp.linalg.norm(pgrid[ind]-jnp.array([0,0,zcross])))
            def eq(data):
                curves = data[0]
                direcs = data[1]
                cind = jnp.array(curves[-1],dtype=int)
                #jax.debug.print("??{x},{y}",x=sing1*r1,y=sing2*r2)
                curves = curves.at[cind].set(1.0/((r1+sing2*r2)/2))
                direcs = direcs.at[cind].set(jnp.arctan2(direc2[1],direc2[0]))
                curves = curves.at[-1].set(cind+1)
                
                return (curves,direcs)
            def neq(data):
                #jax.debug.print("OHON")
                return data
            data  =jax.lax.cond(sing1!=sing2,neq,eq,data)
            return data

        def zero(data):
            return data
        data = jax.lax.cond(jnp.logical_or(jnp.linalg.norm(pgrid[ind]) < 1e-5,jnp.linalg.norm(pgrid[ind]) > 50),zero,nzero,data)
        
        return data,ind
    data,_ = jax.lax.scan(nloop,(curves,direcs),jnp.arange(pgrid.shape[0]))
    curves = data[0]
    direcs = data[1]
    cend = jnp.array(curves[-1],dtype=int)
    return curves,direcs,cend

def set_pbc(pbc,x):
    while(x[0] >pbc[0]):
        x[0] -= pbc[0]
    while(x[0] < 0):
        x[0] += pbc[0]
    while(x[1] >pbc[1]):
        x[1] -= pbc[1]
    while(x[1] < 0):
        x[1] += pbc[1]
    while(x[2] >pbc[2]):
        x[2] -= pbc[2]
    while(x[2] < 0):
        x[2] += pbc[2]
    return x




def load_cg_pdb(file_name, lipid_beads):
    poses = []
    pol_poses = []
    # tail_dir and tail_dir_pt are not needed for APL
    pres = 1
    ppol = False
    head_pos = np.array([0,0,0])
    all_lip_poses = []
    all_lip_red = []
    restype = []

    with open(file_name, "r") as lfile:
        content = lfile.read().split("\n")

    dims = np.array([float(d) for d in content[0].split()[1:4]])

    for c in content:
      if c.startswith("ATOM") and "[" not in c:
            parts = c.split()
            bead = parts[2]
            res = parts[3]
            resi = int(parts[4])
            pos = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
            if not np.any(np.isnan(pos)):
                if bead in ["SC1","SC2","SC3","SC4","SC5","SC6"]:
                    pol_poses.append(pos)
                    ppol = True
                else:
                    if ppol:
                        ppol = False
                        pres = resi
                    rep = False
                    all_rep = False
                    for i in lipid_beads.keys():
                        if res == i:
                            all_rep = True
                            if bead == lipid_beads[i]:
                                rep = True
                                break
                    if rep:
                        head_pos = pos.copy()
                        poses.append(pos)
                        restype.append(res)

                    if all_rep:
                        all_lip_poses.append(pos)
                        if not rep:
                            all_lip_red.append(pos)
            pres = resi

    poses = np.array(poses)
    pol_poses = np.array(pol_poses)
    all_lip_poses = np.array(all_lip_poses)
    all_lip_red = np.array(all_lip_red)

    # tail_dir is now just a placeholder (not needed for APL)
    tail_dir = np.zeros((poses.shape[0], 3))

    return poses, pol_poses, all_lip_poses, dims, restype, all_lip_red, tail_dir

def fix_near_prot(pol_poses,poses,normals):
    cutoff = 15
    close_inds = jnp.zeros(poses.shape[0]+1,dtype=int)
    def norloop(close_inds,ind):
        cposes = center_poses_jax(poses[ind],poses,dims)
        cpol_poses = center_poses_jax(poses[ind],pol_poses,dims)
        isdone = True
        def nor2loop(cdata,ind2):
            close_inds = cdata[0]
            isdone = cdata[1]
            def close(close_inds,isdone):
                cind = jnp.array(close_inds[-1],dtype=int)
                close_inds = close_inds.at[cind].set(ind)
                close_inds = close_inds.at[-1].set(cind+1)
                return close_inds,False
            def far(close_inds,isdone):
                return close_inds,isdone
            close_inds,isdone = jax.lax.cond(jnp.logical_and(jnp.linalg.norm(cpol_poses[ind2]-cposes[ind]) < cutoff,isdone),close,far,close_inds,isdone)
            return (close_inds,isdone),ind2
        cdata,_ = jax.lax.scan(nor2loop,(close_inds,isdone),jnp.arange(pol_poses.shape[0]))
        close_inds = cdata[0]
        return close_inds,ind
    close_inds,_ = jax.lax.scan(norloop,close_inds,jnp.arange(poses.shape[0]))


    normals = replace(normals,close_inds,poses)
    """
    jax.debug.print("{x}",x=close_inds)
    far_inds = jnp.arange(poses.shape[0])
    far_inds = jnp.delete(far_inds,close_inds[:close_inds[-1]])

    def fnploop(normals,ind):
        cposes = center_poses_jax(poses[close_inds[ind]],poses,dims)
        coff = 100000.0
        closest = 0
        def fnp2loop(cdata,ind2):
            closest = cdata[0]
            coff = cdata[1]
            def close(closest,coff):
                closest = ind2
                coff = jnp.linalg.norm(cposes[close_inds[ind]]-cposes[ind2])
                return closest,coff
            def far(closest,coff):
                return closest,coff
            closest,coff = jax.lax.cond(jnp.linalg.norm(cposes[close_inds[ind]]-cposes[ind2]) < coff,close,far,closest,coff)
            return (closest,coff),ind2
        cdata,_ = jax.lax.scan(fnp2loop,(closest,coff),far_inds)
        closest = cdata[0]
        normals = normals.at[close_inds[ind]].set(normals[closest])
        return normals,ind
    normals,_ = jax.lax.scan(fnploop,normals,jnp.arange(close_inds[-1]))
    """
    return normals


def replace(normals,close_inds,poses):
    far_inds = jnp.arange(poses.shape[0])
    far_inds = jnp.delete(far_inds,close_inds)

    def fnploop(normals,ind):
        cposes = center_poses_jax(poses[close_inds[ind]],poses,dims)
        coff = 100000.0
        closest = 0
        def fnp2loop(cdata,ind2):
            closest = cdata[0]
            coff = cdata[1]
            def close(closest,coff):
                closest = ind2
                coff = jnp.linalg.norm(cposes[close_inds[ind]]-cposes[ind2])
                return closest,coff
            def far(closest,coff):
                return closest,coff
            closest,coff = jax.lax.cond(jnp.linalg.norm(cposes[close_inds[ind]]-cposes[ind2]) < coff,close,far,closest,coff)
            return (closest,coff),ind2
        cdata,_ = jax.lax.scan(fnp2loop,(closest,coff),far_inds)
        closest = cdata[0]
        normals = normals.at[close_inds[ind]].set(normals[closest])
        return normals,ind
    normals,_ = jax.lax.scan(fnploop,normals,jnp.arange(close_inds.shape[0]))
    return normals

def fix_nans(normals,poses):
    inds = []
    for i,n in enumerate(normals):
        if(np.any(np.isnan(n))):
            inds.append(i)
    if len(inds)>0:
        normals =replace(jnp.array(normals),jnp.array(inds,dtype=int),jnp.array(poses))
    return normals


timeer=time.time()
print("Starting")
lipid_beads = {}
lip_areas = []
for l in lipids:
    lip_areas.append([])
    lipid_bead = l.split(":")
    lipid_beads[lipid_bead[0]] = lipid_bead[1]
    
lip_reses = list(lipid_beads.keys())
print("Loading PDB...")
poses,pol_poses,all_lip_poses,dims,restypes,all_lip_red,tail_dirs = load_cg_pdb(file_name,lipid_beads)
print("poses shape:", np.shape(poses))
print("Number of poses:", len(poses))
print("Example poses:", poses[:5] if len(poses) > 0 else "EMPTY")
print("Done")
def fill_hole(poses):
    acc_dims = np.max(poses,axis=0)-np.min(poses,axis=0)+2
    bnos = np.array(acc_dims/30,dtype=int)
    grid,_ = allocate_to_grid(poses,acc_dims,bnos,np.min(poses,axis=0)-1)
    grid = grid.tolist()
    add_no = 50000
    new_poses = np.concatenate((poses.copy(),np.zeros((add_no,3))))
    num = poses.shape[0]
    i=0
    while i<add_no:
        print(i)
        a = np.random.randint(0,num)
        indx,indy,indz = put_into_grid3d(new_poses[a],acc_dims,bnos,np.min(poses,axis=0)-1)
        #print(indy,acc_dims[1],new_poses[a],bnos[1])
        #print(indx,acc_dims[0],new_poses[a],bnos[0])
        #print(indz,acc_dims[2],new_poses[a],bnos[2])
        #print([grid[indx+i][indy+j][indz+k] for i in [-1,0,1] for k in [-1,0,1] for j in [-1,0,1]]
        indexl = []
        for i1 in [-1,0,1]:
            for j in [-1,0,1]:
                for k in [-1,0,1]:
                    if(0<indx+i1<bnos[0] and 0<indy+j<bnos[1] and 0<indz+k<bnos[2]):
                        indexl.append(grid[(indx+i1)%bnos[0]][(indy+j)%bnos[1]][(indz+k)%bnos[2]])
        indexl = np.concatenate(indexl)
        indexl = indexl[indexl!=0]
        #print(indexl)
        grid_poses = new_poses[indexl]
        if(len(indexl) > 0):
            b = np.random.randint(0,grid_poses.shape[0])
            if(np.linalg.norm(grid_poses[b]-new_poses[a]) < 30):
                print(grid_poses[b])
                print(new_poses[a])
            #np.ravel(np.array(]
                
                new_poses[num+i] = (grid_poses[b]+new_poses[a])/2
                grid[indx][indy][indz].append(num+i)
                i+=1

    return np.array(new_poses)



print("Calculating normals...")


#tail_dirs =smooth_loop(poses,tail_dirs,4,15)
# Only calculate normals if tail_dirs is not empty
# Only calculate normals if tail_dirs is not empty
#if len(tail_dirs) > 0:
 #   normals = get_normalsv3(poses, tail_dirs)
    # Only smooth if normals exist
 #   normals = smooth_loop(poses, normals, 2, 50)
#else:

#normals =  jnp.array(flip_wrong_normals(np.array(poses),np.array(normals)))
#normals = fix_near_prot(jnp.array(pol_poses),jnp.array(poses),normals)
#normals = jnp.array(fix_nans(np.array(normals),poses))


normals = np.zeros((poses.shape[0], 3))
print("Done")
if code_type == "Curv":
    print("Calculating Curvature...")
else:
    print("Calculating APL...")

normals = np.array(normals)

# ------------------------------
# Safe handling of all_lip_poses
# ------------------------------
x_max = np.max(poses[:,0]) + 1
x_min = np.min(poses[:,0]) - 1
y_max = np.max(poses[:,1]) + 1
y_min = np.min(poses[:,1]) - 1
z_max = np.max(poses[:,2]) + 1
z_min = np.min(poses[:,2]) - 1
if(pol_poses.shape[0] == 0):
    pol_poses=np.array([poses[0]])

z_max2 = np.max(pol_poses[:,2])
z_min2 = np.min(pol_poses[:,2])
z_max = max(z_max,z_max2)+1
z_min = min(z_min,z_min2)-1

starts = np.array([x_min,y_min,z_min])
exts = np.array([x_max-x_min,y_max-y_min,z_max-z_min])
cutoff = 50
max_p = 150
ns = np.array(np.floor(exts/cutoff)+1,dtype=int)

vals = np.zeros(poses.shape[0])
#curvs = np.zeros(poses.shape[0])
pperc = -1

for i,p in enumerate(poses):
    perc = int(i*100/poses.shape[0])
    if(perc%10 == 0 and perc != pperc):
        print(str(perc)+"%")
        pperc = perc
    n_total = 0
    norm = np.zeros(3)
    cen_normal = normals[i]
    cpol_poses = center_poses(p,pol_poses,dims)
    grid_pol_poses,_ = get_box_slice(np.zeros_like(cpol_poses),cpol_poses,dims/2,cutoff)
    



    cposes = center_poses(p,poses,dims)
    grid_poses,grid_normals = get_box_slice(normals,cposes,dims/2,cutoff)

    normy = normals[i]
    #normy /= np.linalg.norm(normy)
    rot_mat = get_rot_mat(normy,np.array([0,0,1]))
    
    if(grid_pol_poses.shape[0] >0):
        rot_pol_poses = np.dot(rot_mat,(grid_pol_poses-dims/2).T).T+dims/2
        
        rot_pol_poses = rot_pol_poses[rot_pol_poses[:,2]<dims[2]/2+2]
        rot_pol_poses = rot_pol_poses[rot_pol_poses[:,2]>dims[2]/2-2]

        #write_point(np.concatenate((rot_pol_poses,grid_poses)),restypes,np.zeros(np.concatenate((rot_pol_poses,grid_poses)).shape[0]),"Test_pol.pdb","")

        #exit()
    else:
        rot_pol_poses = grid_pol_poses
    

    #grid_poses = grid_poses[np.dot(grid_normals,cen_normal) > 0]

    if(grid_poses.size < 4 or code_type == "DEBUG"):
        if(np.isnan(normals[i][0])):
            print(i)
            print(normals[i][0])
        vals[i] = normals[i][0]
    else:
        rot_poses = np.dot(rot_mat,(grid_poses-dims/2).T).T+dims/2
        rot_normals = np.dot(rot_mat,(grid_normals).T).T
        if code_type == "Curv":
            
        
            curves,direcs,cend = calc_curvature(jnp.array(rot_poses-dims/2),jnp.array(rot_normals))

            
            curves = np.array(curves[:cend])
            direcs = np.array(direcs[:cend])

            if cend == 0:
                vals[i] = 0
            else:
                kernel = ExpSineSquared(length_scale=2.0,periodicity=np.pi,periodicity_bounds="fixed",length_scale_bounds="fixed") + WhiteKernel()

                gpr = GaussianProcessRegressor(kernel=kernel,

                        random_state=0,normalize_y=True).fit(direcs.reshape(-1,1), curves)

                #print("Score",gpr.score(direcs.reshape(-1,1), curves))

                xs = np.linspace(-np.pi,np.pi,100)

                preds = gpr.predict(xs.reshape(-1,1), return_std=True)[0]


                #plt.plot(xs,preds)
                #plt.scatter(direcs,curves)
                #plt.show()
                k1 = np.min(preds)
                k2 = np.max(preds)
                #print("Curvature",100*(k1+k2)/2.0)
                
                vals[i] = 100*(k1+k2)/2.0
        else:
            for pi in range(rot_poses.shape[0]):
                testpos = rot_poses[pi]-dims/2
                xy_direc = testpos[:2].copy()
                if(np.linalg.norm(testpos) > 1e-5):
                    xy_direc /= np.linalg.norm(xy_direc)
                    acc_len = np.linalg.norm(testpos)
                    
                    rot_poses[pi] = np.array([xy_direc[0],xy_direc[1],0])*acc_len+dims/2



            

            pgrid = np.concatenate([rot_pol_poses[:,:2],rot_poses[:,:2]])
            if(grid_pol_poses.shape[0] >0):
                pgrid_acc = np.concatenate([grid_pol_poses,grid_poses])
            else:
                pgrid_acc = grid_poses
            pgrid = jnp.array(pgrid-dims[:2]/2)
            
            
            order = jnp.argsort(jnp.linalg.norm(pgrid,axis=1))
            pgrid = pgrid[order]
            pgrid = pgrid[:max_p]
                
            pa = np.array([0,4*cutoff])
            pb = np.array([-4*cutoff,-cutoff])
            pc = np.array([4*cutoff,-cutoff])
            
            triangles_jax = jnp.zeros((pgrid.shape[0]+1,3,2))
            triangles_jax = add_triangle(triangles_jax,jnp.array([pa,pb,jnp.zeros(2)]),0)
            triangles_jax = triangles_jax.at[-1].set(1)
            triangles_jax = add_triangle(triangles_jax,jnp.array([pa,jnp.zeros(2),pc]),1)
            triangles_jax = triangles_jax.at[-1].set(2)
            triangles_jax = add_triangle(triangles_jax,jnp.array([jnp.zeros(2),pb,pc]),2)
            triangles_jax = triangles_jax.at[-1].set(3)           
            
            
            
            
            triangles_jax = get_delaunay_cen_jax(triangles_jax,pgrid).block_until_ready()
            inder = int(triangles_jax[-1][0][0])
            tris = triangles_jax[:inder]
            vals[i] = calc_voronoi(tris)
    lip_areas[lip_reses.index(restypes[i])].append(vals[i])
   
    
rader = np.mean(vals)
rader_std = np.std(vals)


print("Time:",time.time()-timeer)
write_point(poses,restypes,vals,outfn,"")

if code_type != "Curv":
    print("Average area (Angstroms^2) per lipid:",rader,"+/-",rader_std)
    for n,l in enumerate(lip_reses):
        print("Average area (Angstroms^2) per", l,np.mean(lip_areas[n]),"+/-",np.std(lip_areas[n])) 
else:
    print("Average mean curvature (nm^-1):",rader,"+/-",rader_std)
    for n,l in enumerate(lip_reses):
        print("Average mean curvature (nm^-1) for", l,np.mean(lip_areas[n]),"+/-",np.std(lip_areas[n])) 


    
    
    
   
       
   
   




    
    









    
