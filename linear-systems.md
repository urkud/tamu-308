# Systems of linear ODEs

In this chapter we discuss how to solve a *system* of linear
differential equations with constant coefficients like this one.

```{math}
x'&=2x+3y;\\
y'&=3x+2y.
```

To reduce dependency on linear algebra, we restrict our considerations
to the case of two unknown functions (or, in other terms, to the case
of an unknown vector function of dimension two). While most of the
content generalizes to any dimension, some tricks below are specific
to dimension two.

## Linear algebra

In this section we briefly explain concepts from linear algebra that
we need to solve systems of ODEs.
 
### Basic operations on matrices

```{prf:definition}
:label: def:matrix

A *matrix* is a table of numbers. We will deal with 2×2 matrices
$A=\begin{pmatrix}a&b\\c&d\end{pmatrix}$ and vectors written as column
matrices $v=\begin{pmatrix}x\\y\end{pmatrix}$.

The *identity matrix* is the matrix
$I=\begin{pmatrix}1&0\\0&1\end{pmatrix}$.
```

Addition, subtraction, and multiplication by a number are define
elementwise. E.g.,

```{math}
\begin{pmatrix}2&1\\3&5\end{pmatrix}+
\begin{pmatrix}3&4\\-3&2\end{pmatrix}=
\begin{pmatrix}5&5\\0&7\end{pmatrix}.
```

```python
from sympy import *
init_printing()
A = Matrix([[2, 1], [3, 5]])
B = Matrix([[3, 4], [-3, 2]])
eye(2), A, B, A + B, A - B, 2 * A + 3 * B
```

Multiplication of a matrix by a vector is defined by the formula

```{math}
\begin{pmatrix}a&b\\c&d\end{pmatrix}
\begin{pmatrix}x\\y\end{pmatrix}=
\begin{pmatrix}ax+by\\cx+dy\end{pmatrix}.
```

```python
from sympy import *
init_printing()
A = Matrix([[2, 1], [3, 5]])
v = Matrix([1, 2])
A * v
```

So, a linear system

```{math}
x'&=ax+by\\
y'&=cx+dy
```

can be written as $v'=Av$, where
$v=\begin{pmatrix}x\\y\end{pmatrix}$,
$A=\begin{pmatrix}a&b\\c&d\end{pmatrix}$.

### Eigenvalues and eigenvectors

Eigenvalues and eigenvectors defined below play a crucial role in
solving a system of linear ODEs.

```{prf:definition}
A nonzero vector $v$ is called an *eigenvector* of a matrix
$A$ with an eigenvalue $λ$ if $Av=λv$.
```

```{note}
The definition requires $v≠0$ but it allows $λ=0$.
```

```{prf:example}
The matrix $A=\begin{pmatrix}2&3\\3&2\end{pmatrix}$ has an eigenvector
$\begin{pmatrix}1\\1\end{pmatrix}$ with eigenvalue $5$ and an
eigenvector $\begin{pmatrix}1\\-1\end{pmatrix}$ with eigenvalue $-1$.
```

```python
from sympy import *
init_printing()
A = Matrix([[2, 3], [3, 2]])
A.eigenvals(), A.eigenvects()
```

Eigenvalues are important for us because of the following theorem.

```{prf:theorem}
:label: eigenvect-solutions

If $Av=λv$, then $x(t)=e^{λt}v$ is a solution of the ODE $x'=Ax$. If
$v₁$ and $v₂$ are two eigenvectors of a $2×2$ matrix $A$ with
eigenvalues $λ₁≠λ₂$, then $C₁e^{λ₁t}v₁+C₂e^{λ₂t}v₂$ is a general
solution of $x'=Ax$.
```

The first statement can be verified by a direct computation. Indeed,
if $Av=λv$, then
```{math}
(e^{λt}v)'=λe^{λt}v=e^{λt}Av=A(e^{λt}v).
```
The other statement can be deduced from a version of the existence and
uniqueness theorem.

To find the eigenvalues of a matrix, we introduce the following
definition.

```{prf:definition}
The *determinant* of a matrix $A=\begin{pmatrix}a&b\\c&d\end{pmatrix}$
is the number $\det A=ad-bc$.
```

````{note}
In higher dimension, the formula for $\det A$ is more complicated.
E.g., in dimension $3$ we have
```{math}
\det\begin{pmatrix}x₁&x₂&x₃\\y₁&y₂&y₃\\z₁&z₂&z₃\end{pmatrix}=
x₁y₂z₃+x₂y₃z₁+x₃y₁z₂-x₃y₂z₁-x₂y₁z₃-x₁y₃z₂.
```
````

```python
from sympy import *
init_printing()

var('a b c d')
Matrix([[a, b], [c, d]]).det()
Matrix([[a, b], [c, d]]).charpoly().as_expr()
```

If a matrix $A$ has a nonzero determinant, then the linear system
$Ax=y$ has a unique solution for all $y$. In particular, $Ax=0$ holds
true only for $x=0$. This justifies the following theorem.

```{prf:theorem}
Eigenvalues of a matrix $A$ are the solutions of its
*characteristic equation* $\det(A-λI)=0$, where $I$ is
the identity matrix.
```

So, in order to find the eigenvalues and eigenvectors of a matrix, we
need to solve the characteristic equation, then solve $(A-λI)v=0$ for
each eigenvalue we found.

````{prf:example}
:label: ex-distinct-eigenvals

Consider the matrix
$A=\begin{pmatrix}2&3\\3&2\end{pmatrix}$. The characteristic
polynomial is given by

```{math}
\det\begin{pmatrix}2-λ&3\\3&2-λ\end{pmatrix}=(2-λ)²-9=λ²-4λ-5.
```

Its roots are $λ₁=-1$ and $λ₂=5$.

To find an eigenvector corresponding to $λ₁=-1$, we need to
solve
```{math}
\begin{pmatrix}2-(-1)&3\\3&2-(-1)\end{pmatrix}
\begin{pmatrix}a\\b\end{pmatrix}=0.
```
Simplifying $2-(-1)=3$ and using the definition of matrix
multiplication, we get
```{math}
\begin{pmatrix}3a+3b\\3a+3b\end{pmatrix}=0.
```
This vector equation is equivalent to two scalar equations, both of
them are $3a+3b=0$. One of the solutions of this equation is
$a=1$, $b=-1$. So,
$\begin{pmatrix}1\\-1\end{pmatrix}$ is an eigenvector of
$A$ with eigenvalue $-1$. Of course, any other
$\begin{pmatrix}a\\-a\end{pmatrix}$, $a≠0$, is an eigenvector too but
we will need only one of them to solve $\mathbf{x}'=A\mathbf{x}$.

Similarly, one can find that $\begin{pmatrix}1\\1\end{pmatrix}$
is an eigenvector of $A$ with eigenvalue $5$.

These computations and {prf:ref}`eigenvect-solutions` imply that
```{math}
\begin{pmatrix}x\\y\end{pmatrix}=
C₁e^{-t}\begin{pmatrix}1\\-1\end{pmatrix}+
C₂e^{5t}\begin{pmatrix}1\\1\end{pmatrix}
```
is a general solution of the system
```{math}
x'&=2x+3y\\
y'&=3x+2y.
```
This vector solution can be written as two formulas for $x$ and
$y$.
```{math}
x&=C₁e^{-t}+C₂e^{5t}\\
y&=-C₁e^{-t}+C₂e^{5t}.
```
````

## System with distinct real roots

Consider a system of differential equations

```{math}
x'&=ax+by\\
y'&=cx+dy
```

It can be written as $v'=Av$, where
$A=\begin{pmatrix}a&b\\c&d\end{pmatrix}$,
$v=\begin{pmatrix}x\\y\end{pmatrix}$.

In order to solve this system, we first write the matrix $A$ and
find its eigenvalues. If $A$ has two distinct real eigenvalues
(in other words, the discriminant of the characteristic equation is
positive), then {prf:ref}`eigenvect-solutions` can be used to find a
general solution, see {prf:ref}`ex-distinct-eigenvals` for an example.

Once we found the eigenvalues, there is another way to solve the
system. Sometimes it leads to simpler computations, sometimes it does
not. Namely, from the general theory we know that $x(t)$ has the
form $C₁e^{λ₁t}+C₂e^{λ₂t}$. Then we can use the first equation
to find $y$. This approach works unless the coefficient
$b$ is equal to zero.

```{prf:example}
Let us solve the same system as in {prf:ref}`ex-distinct-eigenvals`
using the approach described above. Put $x=C₁e^{-t}+C₂e^{5t}$.
Then $y=\frac{x'-2x}{3}=-C₁e^{-t}+C₂e^{5t}$.
```

## Complex eigenvalues

If the matrix of a differential equation has complex eigenvalues, then
{prf:ref}`eigenvect-solutions` yields a formula that uses the complex
exponential function. While this formula is correct, it is natural to
require that the answer is a problem without complex numbers does not
use complex numbers unless necessary.

In this section we explain how to get an answer in terms of sines and
cosines instead of complex exponential. As in the case of distinct
real roots, there are at least two approaches.

```{tab} Eigenvectors
Consider a linear system of ODEs $x'=Ax$, where $A$ is a
$(2×2)$-matrix. If $A$ has eigenvalues $λ±iμ$ and
$\mathbf{a}+i\mathbf{b}$ is a complex eigenvector with eigenvalue
$λ+iμ$, then

$$
\mathbf{u}(t)&=e^{λt}(\mathbf{a}\cos(μt)-\mathbf{b}\sin(μt)),\\
\mathbf{v}(t)&=e^{λt}(\mathbf{a}\sin(μt)+\mathbf{b}\cos(μt))
$$ (eqn:complex-eigenvals-solution)

are solutions of $x'=Ax$ and any solution can be written as
$C₁\mathbf{u}(t)+C₂\mathbf{v}(t)$.
```
```{tab} Alternative method
Consider a linear system of ODEs $x'=Ax$, where $A$ is a
$(2×2)$-matrix with complex eigenvalues $λ±iμ$. From general theory
(see the other method), we know that the first coordinate of the
answer is given by $x(t)=e^{λt}(C₁\cos(μt)+C₂\sin(μt))$. Then we
substitute this formula into $x'=ax+by$ and find $y=\frac{x'-ax}{b}$.
These $x$ and $y$ will automatically satisfy the second equation but
it is usually a good idea to verify this fact.
```

As an example, consider the system

```{math}
x'&=2x-y;\\
y'&=x+2y.
```

The corresponding matrix $A=\begin{pmatrix}2&-1\\1&2\end{pmatrix}$ has
characteristic polynomial $λ²-4λ+5$ and eigenvalues
$λ_{1,2}=2±i$.

```{tab} Eigenvectors
First, we find an eigenvector for the eigenvalue $λ=2+i$. We have

$$
(A-λI)=\begin{pmatrix}-i&-1\\1&-i\end{pmatrix},
$$

hence $(A-λI)\begin{pmatrix}k\\l\end{pmatrix}=0$ is equivalent to
$-ik-l=0$. To avoid division by complex numbers[^compl-division], take
$k=1$, then $l=-i$. So, $v=\begin{pmatrix}1\\-i\end{pmatrix}$ is an
eigenvector of $A$ with eigenvalue $2+i$.

In terms of {ref}`eqn:complex-eigenvals-solution`, we have $λ=2$,
$μ=1$, $\mathbf{a}=\begin{pmatrix}1\\0\end{pmatrix}$,
$\mathbf{b}=\begin{pmatrix}0\\-1\end{pmatrix}$. Then the general
formula gives implies that

$$
\begin{pmatrix}x(t)\\y(t)\end{pmatrix}&=C₁e^{λt}(\mathbf{a}\cos(μt)-\mathbf{b}\sin(μt))+C₂e^{λt}(\mathbf{a}\sin(μt)+\mathbf{b}\cos(μt))\\
&=C₁e^{2t}\begin{pmatrix}\cos(t)\\\sin(t)\end{pmatrix}+C₂e^{2t}\begin{pmatrix}\sin(t)\\-\cos(t)\end{pmatrix}.
$$

In other words,

$$
x(t)&=e^{2t}(C₁\cos t+C₂\sin t)\\
y(t)&=e^{2t}(C₁\sin t-C₂\cos t)
$$

[^compl-division]: There is nothing wrong about division of complex
numbers (e.g., it has all the usual properties of division of real
numbers). To compute $\frac{a+ib}{c+id}$, you multiply both numerator
and denominator by $c-id$. However, it is easy to make a mistake if
you're not very familiar with complex numbers.
```
```{tab} Alternative method
Take $x(t)=e^{2t}(C₁\cos t+C₂\sin t)$. Then
$x'=2e^{2t}(C₁\cos t+C₂\sin t)+e^{2t}(-C₁\sin t+C₂\cos t)$,
$y=2x-x'=e^{2t}(C₁\sin t-C₂\cos t)$.
```

## Repeated eigenvalues

If $A$ has a repeated eigenvalue $λ$, then one can use the
"alternative" method of solution with $x=C₁e^{λt}+C₂te^{λt}$

```{todo}
Expand, add an example, explain how to solve using eigenvalues and
eigenvectors.
```

## Fundamental matrix of solutions

## Matrix exponential
