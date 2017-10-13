      subroutine f_f77(neq, t, X, Y)
Cf2py intent(hide) neq
Cf2py intent(out) Y
      integer neq
      double precision t, X, Y
      dimension X(neq), Y(neq)
      Y(1)=1.0d0*X(3)
      Y(2)=1.0d0*X(4)/(X(1)*X(5)+1)**2
      Y(3)=-1.0d0*X(1)+1.0d0*X(4)**2*X(5)/(X(1)*X(5)+1)**3
      Y(4)=-1.0d0*X(2)*X(6)**2
      Y(5)=0
      Y(6)=0
      return
      end

