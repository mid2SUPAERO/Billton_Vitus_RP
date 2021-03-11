library(tictoc)
library(PRIMME)
library(rsvd)

n=5000;
A = diag(1:10, 100000,150)
#B = matrix(rexp(n*n, rate=.1), ncol=n)

tic("Primme svd time")
r1 <- PRIMME::svds(A,50)
print("svd complete")
toc(quiet=FALSE)

tic("rsvd time")
r2 <- rsvd(A, k=50)
print("rsvd complete")
toc(quiet = FALSE)


