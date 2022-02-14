
def rot_mb_to_sb(rot):
    ret=np.array()

def rot_sb_to_mb(rot):
    evs,evv=la.eigh(rot+rot.T.conj())
    ev=np.diag(evv.T.conj()@(rot@evv))
    eva=np.angle(ev)
    quad_m=quad_sb_to_mb(evv.T.conj()@eva@evv)
    evm,evvm=la.eigh()
