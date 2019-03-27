
function rig_qltail(Z,A,B,d,e)
    X = [Z A B; 0 d e]; 
    ql!(X)
    d2,e2 = X[1,1] ∩ d, X[1,2] ∩ e
    if isempty(d2) || isempty(e2) || abs(d2) < 1000eps() || abs(e2) < 1000eps()
        return emptyinterval(),emptyinterval()
    end
    len(d2) < 100eps() && len(e2) < 100eps() && return d,e
    if d2 == d && e2 == e
        dm,em = mid(d),mid(e)
        d,e = rig_qltail(Z,A,B,Interval(d2.lo,dm),Interval(e2.lo,em))
        !isempty(d) && !isempty(e) && return d,e
        d,e = rig_qltail(Z,A,B,Interval(dm,d2.hi),Interval(e2.lo,em))
        !isempty(d) && !isempty(e) && return d,e
        d,e = rig_qltail(Z,A,B,Interval(d2.lo,dm),Interval(em,e2.hi))
        !isempty(d) && !isempty(e) && return d,e
        d,e = rig_qltail(Z,A,B,Interval(dm,d2.hi),Interval(em,e2.hi))
        return d,e
    else
        rig_qltail(Z,A,B,d2,e2)
    end
end

function rig_qltail(Z,A,B)
    M = 2maximum(abs,(Z,A,B))
    d,e = Interval(-M,M), Interval(-M,prevfloat(0.0))
    d,e = rig_qltail(Z,A,B,d,e)
    if isempty(d) || isempty(e) 
        d,e = Interval(-M,M), Interval(nextfloat(0.0),M)
        d,e = rig_qltail(Z,A,B,d,e)
    end
    X = [Z A B; 0 d e]; 
    ql!(X)
    X[1,1] ∩ d, X[1,2] ∩ e
end