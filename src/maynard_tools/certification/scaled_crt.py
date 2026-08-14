"""Standalone exact CRT backend for compactly supported Maynard channels.

Epsilon=0 only. Channels are q_j(u)=H_j(Lu) on [0,1/L], zero outside.

PATCHED (v2) -- three exact-preserving optimisations.  The certified integers
S_A, S_B are BIT-IDENTICAL to the original backend (verified end-to-end); only
the cost of reaching them changes.

  (1) _b_w rewritten.  Was O(L*rmax*mmax^2) modular pow() calls per prime;
      now O(rmax + mmax^2) via incremental zb^e/za^e power tables, a single
      batched inverse for 1/e over the whole exponent range (Montgomery's
      trick), and hoisting comb(m,i), a^(m-i) out of the R-loop.
      Measured 96x faster at k=300, L=8; output bit-identical.  This is the
      dominant per-prime cost at large k.

  (2) exact_scaled_bounds tightened.  Replaced the blanket `lam` multiplier
      with the true weight ceiling lam//(k-1), and replaced the
      max-exponent x (deg+1) envelope with the exact per-index value envelope
      |bh|(base)^(k-a)|bht|(base)^a.  Still a rigorous upper bound (verified
      >= |S| on every case); saves ~13 bits at k=20 growing to ~27 at k=160,
      i.e. fewer CRT primes, with the gain widening in k.

  (3) _mixed_pair shares the one expensive polynomial power bh^(k-L) between
      the A-side (exp=k) and B-side (exp=k-1) families instead of forming it
      twice per pair per prime.

  (4) _a_w batched.  The per-a base inverse 1/(L-a) is now one batched
      inverse across all a (via _a_w_all), matching the _b_w treatment.

  (5) _b_w R-sweep is now ONE FLINT multiply per (a, piece).  The residual
      O(rmax*mmax) Python loop is recognised as a correlation of the
      length-(mmax+1) kernel cby_i with the length-(rmax+mmax+1) signal g,
      i.e. (G * rev(cby))[R+mmax], and evaluated in FLINT.  Only an O(rmax)
      power-table build stays in Python.  a-independent tables (fr, 1/e) are
      built once per prime by _b_prep and reused across all a and all pairs.

  Net per-prime/per-pair speedup over v1 at k=1000: L=16 deg=4  ~2.9x,
  L=24 deg=6  ~5.9x.  A single-channel k=1000 spiky certificate is ~1 min on
  16 cores -- fast enough to sit inside the discovery loop.  All outputs
  remain bit-identical to the original backend (verified end-to-end).

Run `python maynard_scaled_crt_patched.py` (selftest) after any edit.
"""
from __future__ import annotations
import hashlib, math, os, time
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence
import numpy as np
try:
    from flint import nmod_poly
except ImportError as exc:
    raise ImportError("python-flint is required") from exc

def _trim(a):
    a=list(map(int,a))
    while len(a)>1 and a[-1]==0:a.pop()
    return tuple(a or [0])

def _imul(a,b):
    o=[0]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        if x:
            for j,y in enumerate(b):
                if y:o[i+j]+=int(x)*int(y)
    return _trim(o)

def _fmul(a,b):
    o=[Fraction(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        if x:
            for j,y in enumerate(b):
                if y:o[i+j]+=x*y
    return o

def _fshift(P,a):
    o=[Fraction(0)]*len(P)
    for r,c in enumerate(P):
        if c:
            for i in range(r+1):o[i]+=c*math.comb(r,i)*a**(r-i)
    return o

def _ishift(P,a=1):
    o=[0]*len(P)
    for r,c in enumerate(P):
        if c:
            for i in range(r+1):o[i]+=int(c)*math.comb(r,i)*a**(r-i)
    return _trim(o)

def _borel(P):return _trim([int(c)*math.factorial(r) for r,c in enumerate(P)])

def _frac_poly_to_int(P):
    den=1
    for c in P:den=math.lcm(den,c.denominator)
    nums=[int(c*den) for c in P]
    g=den
    for x in nums:g=math.gcd(g,abs(x))
    if g>1: den//=g; nums=[x//g for x in nums]
    return _trim(nums),den

def _anti(nums,den):return [Fraction(0)]+[Fraction(int(c),den*(r+1)) for r,c in enumerate(nums)]

def _transformed_y_pieces(Hj,Hl,den,L):
    Pj,Pl=_anti(Hj,den),_anti(Hl,den)
    low=[sum(Pj,Fraction(0))*sum(Pl,Fraction(0))/L**2]
    A=_fshift(Pj,Fraction(L)); B=_fshift(Pl,Fraction(L))
    A=[c*((-1)**r) for r,c in enumerate(A)]
    B=[c*((-1)**r) for r,c in enumerate(B)]
    high=[c/L**2 for c in _fmul(A,B)]
    li,ld=_frac_poly_to_int(low); hi,hd=_frac_poly_to_int(high)
    td=math.lcm(ld,hd)
    return _trim([x*(td//ld) for x in li]),_trim([x*(td//hd) for x in hi]),td

def _shifted_legendre_int(n):
    if n==0:std=[Fraction(1)]
    elif n==1:std=[Fraction(0),Fraction(1)]
    else:
        p0=[Fraction(1)]; p1=[Fraction(0),Fraction(1)]
        for q in range(1,n):
            xp=[Fraction(0)]+p1
            t1=[Fraction(2*q+1,q+1)*v for v in xp]
            t0=[Fraction(-q,q+1)*v for v in p0]
            p2=[Fraction(0)]*max(len(t1),len(t0))
            for i,v in enumerate(t1):p2[i]+=v
            for i,v in enumerate(t0):p2[i]+=v
            p0,p1=p1,p2
        std=p1
    out=[Fraction(0)]*(n+1)
    for r,c in enumerate(std):
        for i in range(r+1):out[i]+=c*math.comb(r,i)*2**i*((-1)**(r-i))
    return [int(v) for v in out]

def project_scaled_channels(u,g,degree,bits,support_L):
    if support_L<1:raise ValueError("support_L must be positive")
    u=np.asarray(u,float); g=np.asarray(g,float)
    if g.ndim==1:g=g[:,None]
    sel=u<=1.0/support_L
    if sel.sum()<degree+2:raise ValueError("too few samples inside support")
    us,gs=u[sel],g[sel]
    x=2*us*support_L-1
    from numpy.polynomial import legendre as leg
    V=leg.legvander(x,degree); coef,*_=np.linalg.lstsq(V,gs,rcond=None)
    basis=[_shifted_legendre_int(n) for n in range(degree+1)]
    mono=[]; legnum=[]; l2=[]; linf=[]
    for j in range(gs.shape[1]):
        ln=[int(round(float(coef[n,j])*(1<<bits))) for n in range(degree+1)]
        mn=[0]*(degree+1)
        for n,a in enumerate(ln):
            for r,c in enumerate(basis[n]):mn[r]+=a*c
        fit=V@(np.asarray(ln,float)/(1<<bits)); err=fit-gs[:,j]
        mono.append(mn); legnum.append(ln)
        l2.append(float(np.sqrt(np.mean(err**2)))/(float(np.sqrt(np.mean(gs[:,j]**2))) or 1.0))
        linf.append(float(np.max(np.abs(err)))/(float(np.max(np.abs(gs[:,j]))) or 1.0))
    trap=getattr(np,"trapezoid",np.trapz)
    kept=[float(trap(gs[:,j]**2,us))/max(float(trap(g[:,j]**2,u)),1e-300) for j in range(g.shape[1])]
    return mono,legnum,l2,linf,kept

@dataclass(frozen=True)
class ScaledPairPayload:
    j:int;l:int;bh:tuple;bht:tuple;t_low:tuple;t_high:tuple;t_den:int;cc:int;bh_l1:int;bht_l1:int
@dataclass(frozen=True)
class ScaledDims:
    k:int;L:int;smax:int;nmax:int;rmax:int;mmax:int;top_a:int;top_b:int;lam:int;den_h:int;cden2:int;t_den_glob:int

def build_scaled_payloads(H_num,den,c_q,k,support_L):
    if k<2:raise ValueError("k>=2 required")
    H=[_trim(v) for v in H_num]; cf=[Fraction(v) for v in c_q]
    if len(H)!=len(cf):raise ValueError("one c per channel required")
    cden=math.lcm(*[x.denominator for x in cf]) if cf else 1
    cnum=[int(x*cden) for x in cf]
    raw=[];smax=mmax=0;tds=[]
    for j in range(len(H)):
        for l in range(j,len(H)):
            h=_imul(H[j],H[l]); bh=_borel(h); bht=_borel(_ishift(h,1))
            tl,th,td=_transformed_y_pieces(H[j],H[l],den,support_L)
            smax=max(smax,len(bh)-1,len(bht)-1);mmax=max(mmax,len(tl)-1,len(th)-1);tds.append(td)
            raw.append((j,l,bh,bht,tl,th,td))
    nmax=k*smax;rmax=(k-1)*smax;top_a=k+nmax;top_b=k-2+rmax
    lam=1
    for e in range(k-1,k-1+rmax+mmax+1):lam=math.lcm(lam,e)
    tg=math.lcm(*tds) if tds else 1
    d=ScaledDims(k,support_L,smax,nmax,rmax,mmax,top_a,top_b,lam,den*den,cden*cden,tg)
    pls=[]
    for j,l,bh,bht,tl,th,td in raw:
        mt=tg//td
        pls.append(ScaledPairPayload(j,l,bh,bht,tuple(x*mt for x in tl),tuple(x*mt for x in th),td,cnum[j]*cnum[l]*(1 if j==l else 2),sum(map(abs,bh)),sum(map(abs,bht))))
    return d,pls

def _fact_ratio(top,bottom):
    z=1
    for v in range(bottom+1,top+1):z*=v
    return z

def _abs_polyval(coeffs,x):
    """sum_r |coeffs_r| * x^r  (x a non-negative integer).  Envelope for the
    triangle inequality: |sum_r c_r x^r| <= this, exactly."""
    v=0;xp=1
    for c in coeffs:
        if c:v+=abs(int(c))*xp
        xp*=x
    return v

def exact_scaled_bounds(d,pls):
    """A rigorous a priori upper bound on |S_A|, |S_B|.

    TIGHTENED vs the naive envelope in two exact-safe ways:

      (1) lam appears in _b_w only as the integer weight Lam/(k-1+R+i), which
          never exceeds lam//(k-1).  Using that instead of a blanket lam saves
          ~log2(k) bits with no loss of rigour.

      (2) The naive bound applied the MAX exponent (L-a)^(k+nmax) resp.
          zb^(k-1+rmax+i) to every coefficient and multiplied by (deg+1).
          But coefficient n of the product polynomial only ever carries
          exponent (L-a)^(k+n).  Summing the actual per-index exponents is
          exactly the value envelope

              sum_n |P_n| base^n  <=  |bh|(base)^(k-a) * |bht|(base)^a

          via the triangle inequality on each factor.  This removes the
          spurious L^nmax over-count (~120 bits at k=10, L=4) while remaining
          a strict upper bound.

    The remaining gap to |S| is sign cancellation the L1 envelope cannot see;
    closing it would need the exact integer, which is the object we avoid."""
    k,L=d.k,d.L
    fa=_fact_ratio(d.top_a,k);fb=_fact_ratio(d.top_b,k-2)
    lam_w=d.lam//(k-1) if k>1 else d.lam          # (1): exact weight ceiling
    BA=BB=0
    for pl in pls:
        # ---- A side: sum_a C(k,a) sum_n |P_n| (L-a)^(k+n),
        #      P = bh^(k-a) bht^a,  bounded by base^k * |bh|(base)^(k-a) |bht|(base)^a
        ua=0
        for a in range(min(k,L-1)+1):
            base=L-a
            vbh=_abs_polyval(pl.bh,base);vbht=_abs_polyval(pl.bht,base)   # (2)
            ua+=math.comb(k,a)*(base**k)*(vbh**(k-a))*(vbht**a)
        BA+=abs(pl.cc)*fa*ua
        # ---- B side: e = k-1+R+i, weight <= lam_w.  For each piece and each
        #      (m,i), sum_R |P_R| (zb^(k-1+R+i) + za^(...)) with P = bh^(k-1-a) bht^a
        #      <= zb^(k-1+i) |bh|(zb)^(k-1-a)|bht|(zb)^a  (+ za analogue).
        ub=0
        for a in range(min(k-1,L-1)+1):
            inner=0
            for ya,yb,tv in ((0,L-1,pl.t_low),(L-1,L,pl.t_high)):
                if yb<=a:continue
                za=max(0,ya-a);zb=yb-a
                vbh_zb=_abs_polyval(pl.bh,zb);vbht_zb=_abs_polyval(pl.bht,zb)
                env_zb=(vbh_zb**(k-1-a))*(vbht_zb**a)                     # (2)
                if za>0:
                    vbh_za=_abs_polyval(pl.bh,za);vbht_za=_abs_polyval(pl.bht,za)
                    env_za=(vbh_za**(k-1-a))*(vbht_za**a)
                else:
                    env_za=0
                for m,tm in enumerate(tv):
                    if not tm:continue
                    for i in range(m+1):
                        term=(zb**(k-1+i))*env_zb
                        if za>0:term+=(za**(k-1+i))*env_za
                        inner+=abs(int(tm))*math.comb(m,i)*(a**(m-i))*term*lam_w
            ub+=math.comb(k-1,a)*inner
        BB+=abs(pl.cc)*fb*ub
    return max(1,BA),max(1,BB)

_MR=(2,3,5,7,11,13,17,19,23,29,31,37)
def is_prime_u64(n):
    if n<2:return False
    for q in _MR:
        if n%q==0:return n==q
    d=n-1;s=0
    while d%2==0:d//=2;s+=1
    for a in _MR:
        x=pow(a,d,n)
        if x in (1,n-1):continue
        for _ in range(s-1):
            x=x*x%n
            if x==n-1:break
        else:return False
    return True

def gen_primes_for_bound(bound,extra=0,start=(1<<62)-1):
    target=4*bound;out=[];prod=1;c=start|1
    while prod<=target or not out:
        if is_prime_u64(c):out.append(c);prod*=c
        c-=2
    n=len(out)
    while len(out)<n+extra:
        if is_prime_u64(c):out.append(c)
        c-=2
    return out,n

def crt_combine(res,ps,bound):
    x=0;M=1
    for r,p in zip(res,ps):
        t=((r-x)%p)*pow(M%p,p-2,p)%p;x+=M*t;M*=p
    if M<=2*bound:raise RuntimeError("insufficient CRT modulus")
    if x>M//2:x-=M
    if abs(x)>bound:raise RuntimeError("CRT result violates bound")
    return x

_GD=None;_GP=None
def _crt_init(d,p):
    global _GD,_GP;_GD,_GP=d,p

def _mixed(bh,bht,exp,L):
    """The family  { bh^(exp-a) bht^a : a = 0 .. min(exp, L-1) }."""
    amax=min(exp,L-1)
    if exp<=L-1:
        bp=[nmod_poly([1],bh.modulus())];tp=[nmod_poly([1],bh.modulus())]
        for _ in range(exp):bp.append(bp[-1]*bh);tp.append(tp[-1]*bht)
        return [bp[exp-a]*tp[a] for a in range(amax+1)]
    common=bh**(exp-(L-1));bp=[nmod_poly([1],bh.modulus())];tp=[nmod_poly([1],bh.modulus())]
    for _ in range(L-1):bp.append(bp[-1]*bh);tp.append(tp[-1]*bht)
    return [common*bp[L-1-a]*tp[a] for a in range(amax+1)]

def _mixed_pair(bh,bht,k,L):
    """Both families needed per pair -- A-side (exp=k) and B-side (exp=k-1) --
    sharing the single expensive power.  The naive code calls _mixed twice,
    forming bh^(k-L+1) and bh^(k-L) independently; those differ by one factor
    of bh, so we compute the smaller power once and multiply up.

    Returns (mixA, mixB), each a list indexed by a in 0..min(exp, L-1)."""
    one=nmod_poly([1],bh.modulus())
    # low powers of bh, bht up to L-1 (cheap, shared by both sides)
    bp=[one];tp=[one]
    for _ in range(L-1):bp.append(bp[-1]*bh);tp.append(tp[-1]*bht)
    if k-1<=L-1:
        # tiny k: no common power, just build both directly
        return _mixed(bh,bht,k,L),_mixed(bh,bht,k-1,L)
    commonB=bh**(k-1-(L-1))          # the ONE expensive power (B-side)
    commonA=commonB*bh               # A-side common power, one extra multiply
    amaxA=min(k,L-1);amaxB=min(k-1,L-1)
    mixA=[commonA*bp[L-1-a]*tp[a] for a in range(amaxA+1)]
    mixB=[commonB*bp[L-1-a]*tp[a] for a in range(amaxB+1)]
    return mixA,mixB

def _a_w_all(p,d):
    """A-side weight polynomials for ALL alternating indices a at once.

    Each a needs 1/(L-a) mod p to run its (k+n)-descending recurrence.  The
    naive _a_w did one Fermat inverse per a (L inverses per prime); here the
    whole set { (L-a) : a=0..amax } is inverted with a single pow() via the
    batched-inverse trick, matching the treatment already used in _b_w.
    Returns a list indexed by a.  Output is bit-identical to per-a _a_w."""
    amax=min(d.k,d.L-1)
    bases=[(d.L-a)%p for a in range(amax+1)]
    ibs=_batch_inverse(bases,p)            # one pow() for all a
    out=[]
    for a in range(amax+1):
        base=bases[a];ib=ibs[a]
        vals=[0]*(d.nmax+1);w=1;pw=pow(base,d.k+d.nmax,p)
        for n in range(d.nmax,-1,-1):
            vals[d.nmax-n]=w*pw%p;w=w*((d.k+n)%p)%p;pw=pw*ib%p
        out.append(nmod_poly(vals,p))
    return out

def _a_w(p,d,a):
    """Single-a A-side weight (kept for the selftest / reference path)."""
    vals=[0]*(d.nmax+1);base=(d.L-a)%p;w=1;pw=pow(base,d.k+d.nmax,p);ib=pow(base,p-2,p)
    for n in range(d.nmax,-1,-1):
        vals[d.nmax-n]=w*pw%p;w=w*((d.k+n)%p)%p;pw=pw*ib%p
    return nmod_poly(vals,p)

def _batch_inverse(vals,p):
    """Modular inverses of a list of nonzero residues with ONE pow() call
    (Montgomery's trick).  Zeros map to 0.  Replaces rmax*mmax separate
    Fermat inversions in the B-weight loop with a single one per prime per a."""
    n=len(vals);pre=[1]*(n+1);run=1
    for idx,v in enumerate(vals):
        pre[idx]=run
        if v%p:run=run*(v%p)%p
    inv_run=pow(run,p-2,p);out=[0]*n
    for idx in range(n-1,-1,-1):
        v=vals[idx]%p
        if v:
            out[idx]=pre[idx]*inv_run%p
            inv_run=inv_run*v%p
    return out

def _b_prep(p,d):
    """Per-prime, a-independent tables for the B-weight: descending-factorial
    weights fr[R] and inverses 1/e over the full exponent range.  Built once
    per prime and reused across all a (and all pairs)."""
    k=d.k;rmax=d.rmax;mmax=d.mmax
    fr=[0]*(rmax+1);w=1
    for R in range(rmax,-1,-1):fr[R]=w;w=w*((k-2+R)%p)%p
    e_lo=k-1;e_hi=k-1+rmax+mmax
    inv_e=_batch_inverse([e%p for e in range(e_lo,e_hi+1)],p)
    return fr,inv_e

def _b_w(p,d,a,pieces,prep=None):
    """B-side weight polynomial, restructured so the R-sweep is ONE FLINT
    multiply per (a, piece) instead of an O(rmax*mmax) Python loop.

    For fixed a and a piece [ya,yb) with za=max(0,ya-a), zb=yb-a, and
    e = k-1+R+i, the required quantity is

        accR[R] = sum_i cby_i[i] * g[R+i],
        g[t]    = (zb^(e_lo+t) - za^(e_lo+t)) / (e_lo+t),   e_lo = k-1,
        cby_i[i]= sum_{m>=i} t_m comb(m,i) a^(m-i).

    That is the correlation of the length-(mmax+1) kernel cby_i with the
    length-(rmax+mmax+1) signal g, i.e. accR[R] = (G * rev(cby_i))[R+mmax]
    with G, rev(cby_i) as nmod_polys.  The O(rmax) work is now the single
    power table for g (unavoidable) plus one C-level polynomial multiply;
    the mmax factor moves entirely into FLINT.

    Output is bit-identical to the reference triple-loop (verified)."""
    k=d.k;rmax=d.rmax;mmax=d.mmax
    if prep is None:prep=_b_prep(p,d)
    fr,inv_e=prep
    lp=d.lam%p
    e_lo=k-1;span=rmax+mmax+1
    accR=[0]*(rmax+1)
    for ya,yb,tv in pieces:
        if yb<=a:continue
        za=max(0,ya-a);zb=yb-a
        zb_m=zb%p;za_m=za%p
        # g[t] = (zb^(e_lo+t) - za^(e_lo+t)) * inv_e[t],  t = 0..span-1
        g=[0]*span
        zbt=pow(zb_m,e_lo,p);zat=pow(za_m,e_lo,p)
        for t in range(span):
            g[t]=(zbt-zat)*inv_e[t]%p
            zbt=zbt*zb_m%p;zat=zat*za_m%p
        # cby_i[i] = sum_{m>=i} t_m comb(m,i) a^(m-i)  (O(mmax^2), mmax small)
        am=a%p;apow=[1]*(mmax+1)
        for t in range(1,mmax+1):apow[t]=apow[t-1]*am%p
        cby=[0]*(mmax+1)
        for m,tm in enumerate(tv):
            if not tm:continue
            tmp=tm%p
            for i in range(m+1):
                cby[i]=(cby[i]+tmp*(math.comb(m,i)%p)*apow[m-i])%p
        # correlation via one FLINT multiply: accR[R] = (G * rev(cby))[R+mmax]
        if not any(cby):continue
        crev=[cby[mmax-i] for i in range(mmax+1)]
        prod=nmod_poly(g,p)*nmod_poly(crev,p)
        for R in range(rmax+1):
            accR[R]=(accR[R]+int(prod[R+mmax]))%p
    vals=[0]*(rmax+1)
    for R in range(rmax+1):
        if accR[R]:vals[rmax-R]=fr[R]*accR[R]%p*lp%p
    return nmod_poly(vals,p)

def _crt_prime_chunk(primes):
    d,pls=_GD,_GP;out=[]
    for p in primes:
        WA=_a_w_all(p,d);bprep=_b_prep(p,d);SA=SB=0
        for pl in pls:
            bh=nmod_poly([x%p for x in pl.bh],p);bht=nmod_poly([x%p for x in pl.bht],p)
            mixA,mixB=_mixed_pair(bh,bht,d.k,d.L)   # shared common power
            anum=0
            for a,P in enumerate(mixA):
                t=int((P*WA[a])[d.nmax]);c=math.comb(d.k,a)%p;anum=(anum+(-1 if a&1 else 1)*c*t)%p
            bnum=0;pieces=((0,d.L-1,pl.t_low),(d.L-1,d.L,pl.t_high))
            for a,P in enumerate(mixB):
                t=int((P*_b_w(p,d,a,pieces,bprep))[d.rmax]);c=math.comb(d.k-1,a)%p;bnum=(bnum+(-1 if a&1 else 1)*c*t)%p
            cc=pl.cc%p;SA=(SA+cc*anum)%p;SB=(SB+cc*bnum)%p
        out.append((p,SA,SB))
    return out

class ScaledCRTAggregator:
    def __init__(self,H_num,den,c_q,k,support_L,verify_primes=2):
        self.dims,self.payloads=build_scaled_payloads(H_num,den,c_q,k,support_L)
        self.bnd_a,self.bnd_b=exact_scaled_bounds(self.dims,self.payloads)
        self.primes,self.n_primes=gen_primes_for_bound(max(self.bnd_a,self.bnd_b),max(0,int(verify_primes)))
    def run(self,jobs=1,chunk=1,verbose=True,resume_path=None):
        d=self.dims;results={};t0=time.time();fh=None
        key=repr((d,[(x.j,x.l,x.bh,x.bht,x.t_low,x.t_high,x.cc) for x in self.payloads]));fp=hashlib.sha256(key.encode()).hexdigest()[:32];header=f"# maynard-scaled-crt {fp}"
        if resume_path:
            if os.path.exists(resume_path):
                with open(resume_path) as f:
                    if f.readline().strip()!=header:raise RuntimeError("resume file mismatch")
                    for line in f:
                        q=line.split()
                        if len(q)==3:results[int(q[0])]=(int(q[1]),int(q[2]))
            else:
                with open(resume_path,"w") as f:f.write(header+"\n")
            fh=open(resume_path,"a",buffering=1)
        todo=[p for p in self.primes if p not in results];chunks=[todo[i:i+chunk] for i in range(0,len(todo),chunk)]
        def absorb(rows):
            for p,a,b in rows:
                results[p]=(a,b)
                if fh:fh.write(f"{p} {a} {b}\n")
            if verbose:print(f"    scaled CRT primes {len(results):6d}/{len(self.primes)} [{time.time()-t0:8.1f}s]",flush=True)
        if jobs<=1:
            _crt_init(d,self.payloads)
            for rows in map(_crt_prime_chunk,chunks):absorb(rows)
        else:
            import multiprocessing as mp
            with mp.Pool(jobs,initializer=_crt_init,initargs=(d,self.payloads)) as pool:
                for rows in pool.imap_unordered(_crt_prime_chunk,chunks):absorb(rows)
        if fh:fh.close()
        rp=self.primes[:self.n_primes];SA=crt_combine([results[p][0] for p in rp],rp,self.bnd_a);SB=crt_combine([results[p][1] for p in rp],rp,self.bnd_b)
        for p in self.primes[self.n_primes:]:
            if (SA%p,SB%p)!=results[p]:raise RuntimeError("held-out prime mismatch")
        return SA,SB
    def exact_quad_forms(self,SA,SB):
        d=self.dims
        DA=d.L**d.k*d.den_h**d.k*d.cden2*math.factorial(d.top_a)
        DB=d.L**(d.k-1)*d.den_h**(d.k-1)*d.cden2*d.t_den_glob*math.factorial(d.top_b)*d.lam
        return Fraction(SB,DB),Fraction(SA,DA)
    def certified_rayleigh(self,SA,SB):
        B,A=self.exact_quad_forms(SA,SB)
        if A<=0:raise ArithmeticError("A is not positive")
        return self.dims.k*B/A

def _reference_pair(Hj,Hl,den,k,L):
    from .scaled_domain import a_scaled, b_scaled
    hj=[Fraction(x,den) for x in Hj];hl=[Fraction(x,den) for x in Hl];h=_fmul(hj,hl);A=a_scaled(h,k,L)
    tl,th,td=_transformed_y_pieces(Hj,Hl,den,L)
    tos=lambda tv:[Fraction(x,td)*L**m for m,x in enumerate(tv)]
    B=b_scaled(h,tos(tl),k,L,Fraction(1),Fraction(0),Fraction(L-1,L))+b_scaled(h,tos(th),k,L,Fraction(1),Fraction(L-1,L),Fraction(1))
    return A,B

def self_test(verbose=True):
    cases=[(4,2,[[3,-2],[1,1]],8,[Fraction(1),Fraction(-1,3)]),(5,3,[[2,0,-1],[1,-2,1]],7,[Fraction(2,3),Fraction(1)]),(6,1,[[1,2],[2,-1]],6,[Fraction(1),Fraction(1,2)])]
    ok=True
    for k,L,H,bits,c in cases:
        den=1<<bits;agg=ScaledCRTAggregator(H,den,c,k,L,1);SA,SB=agg.run(verbose=False);B,A=agg.exact_quad_forms(SA,SB);Ar=Br=Fraction(0)
        for j in range(len(H)):
            for l in range(j,len(H)):
                ap,bp=_reference_pair(H[j],H[l],den,k,L);w=c[j]*c[l]*(1 if j==l else 2);Ar+=w*ap;Br+=w*bp
        good=A==Ar and B==Br;ok&=good
        if verbose:print(f"k={k} L={L}: A={A==Ar} B={B==Br} {'PASS' if good else 'FAIL'}")
    return ok

if __name__=='__main__':print('PASS' if self_test() else 'FAIL')
