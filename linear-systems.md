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

In terms of {eq}`eqn:complex-eigenvals-solution`, we have $λ=2$,
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

We start with a few examples, then explain the general method.

````{prf:example}
:label: ex-jordan

Consider the linear system

$$
x'&=3x+y\\
y'&=3y
$$

The corresponding matrix $A=\begin{pmatrix}3&1\\0&3\end{pmatrix}$ has
a repeated eigenvalue $3$ with an eigenvector
$\begin{pmatrix}1\\0\end{pmatrix}$. So, {prf:ref}`eigenvect-solutions`
gives us only one solution $x(t)=Ce^{3t}$, $y(t)=0$. This solution is
not enough to find a general solution.

Instead, we can solve the second equation: $y=C₁e^{3t}$, hence the
first equation can be written as $x'-3x=C₁e^{3t}$. This is a linear
equation on $x$ with complementary function $x_c=C₂e^{3t}$. Since
$e^{3t}$ from the RHS appears in the complementary function, a good
guess for a particular solution is $x_p=Ate^{3t}$. We have
$(Ate^{3t})'-3Ate^{3t}=Ae^{3t}$, hence $A=C₁$. Finally,

$$
x&=C₁te^{3t}+C₂e^{3t}\\
y&=C₁e^{3t}.
$$

Similarly to the case of higher order linear equations with repeated
roots of the characteristic polynomial, the formula for a general
solution has terms $e^{3t}$ and $te^{3t}$.
````

````{prf:example} A system with a diagonal matrix
Consider the linear system

$$
x'&=3x\\
y'&=3y
$$

We can solve the equations of this system separately: $x(t)=C₁e^{3t}$,
$y(t)=C₂e^{3t}$.

The corresponding matrix $A=\begin{pmatrix}3&0\\0&3\end{pmatrix}$ has
a repeated eigenvealue $3$. In this case, repeated eigenvalues do not
generate terms with $te^{3t}$ in the answer. In dimension two,

$$
x'&=λx\\
y'&=λy
$$

is the only type of a linear system with repeated eigenvalues but no
$te^{λt}$ in a general soluiton.
````

`````{prf:example}
Consider the system

$$
x'&=-x+4y,\\
y'&=-x+3y.
$$

The corresponding matrix $A=\begin{pmatrix}-1&4\\-1&3\end{pmatrix}$
has a repeated eigenvalue $λ=1$.

````{tab} Eigenvectors
Take a vector $v₁$ that is **not** an eigenvector of $A$. E.g.,
$v₁=\begin{pmatrix}1\\0\end{pmatrix}$. Then we compute

$$
v₂=Av₁-λv₁=\begin{pmatrix}-2\\-1\end{pmatrix}.
$$

Note that $v₂$ is an eigenvector of $A$, $Av₂=v₂$. Thus $e^tv₂$ is a
solution of the original system. Another solution is given by
$e^t(tv₂+v₁)$. Finally,

$$
\begin{pmatrix}x\\y\end{pmatrix}&=C₁e^t(v₁+tv₂)+C₂e^tv₂\\
&=C₁e^t\begin{pmatrix}1-2t\\-t\end{pmatrix}+C₂e^t\begin{pmatrix}-2\\-1\end{pmatrix}
$$

In other words,

$$
x&=C₁e^t(1-2t)-2C₂e^t\\
y&=-C₁te^t-C₂e^t.
$$
````
````{tab} Alternative method
From our experience with higher order differential equations and from
the general theory (see below), $y(t)$ has the form $C₁te^t+C₂e^t$ for
some $C₁$, $C₂$. Then

$$
x&=3y-y'\\
&=3C₁te^t+3C₂e^t-C₁e^t-C₁te^t-C₂e^t\\
&=C₁e^t(2t-1)+2C₂e^t.
$$

This answer differs from the answer we found using eigenvectors by the
change of sign of both $C₁$ and $C₂$.
````
`````

Now we describe the general method. Assume that the matrix $A$ of a
linear sytem $x'=ax+by$, $y'=cx+dy$ has a repeated eigenvalue $λ$. If
$A$ is the diagonal matrix $λI$, then the solution is given by
$x(t)=x(0)e^{λt}$, $y(t)=y(0)e^{λt}$. Otherwise, we can use one of the
following two methods.

````{tab} Eigenvectors
* Choose a vector $v₁$ that is **not** an eigenvector of $A$. Either
  $\begin{pmatrix}1\\0\end{pmatrix}$ or
  $\begin{pmatrix}0\\1\end{pmatrix}$ will work.
* Compute $v₂=Av₁-λv₁$.
* Optional sanity check: $v₂$ is an eigenvector of $A$, $v₂≠0$ and
  $Av₂=λv₂$.
* Write the answer:
  $\begin{pmatrix}x\\y\end{pmatrix}=C₁e^{λt}(v₁+tv₂)+C₂e^{λt}v₂$.
* If required, reformulate the answer as $x=\dots$, $y=\dots$.
````
````{tab} Alternative method
* Put $x=e^{λt}(C₁+C₂t)$.
* Compute $y=\frac{x'-ax}{b}$.

This will only fail if $b=0$. In this case we swap the roles of $x$
and $y$.

* Put $y=e^{λt}(C₁+C₂t)$.
* Compute $x=\frac{y'-dy}{c}$.
````

## Fundamental matrix of solutions

Consider the system of linear ODEs given by

$$
x'&=2x+3y\\
y'&=3x+2y
$$

As we saw in {prf:ref}`ex-distinct-eigenvals`, a general solution of
this system is given by

$$
\begin{pmatrix}x\\y\end{pmatrix}
&=C₁e^{-t}\begin{pmatrix}1\\-1\end{pmatrix}+
C₂e^{5t}\begin{pmatrix}1\\1\end{pmatrix}\\
&=\begin{pmatrix}e^{-t}&e^{5t}\\-e^{-t}&e^{5t}\end{pmatrix}
\begin{pmatrix}C₁\\C₂\end{pmatrix},
$$

where the last equality follows from the definition of matrix
multiplication. This means that *any solution* of this system can be
written as $\mathbf{Ψ}(t)v$, where
$\mathbf{Ψ}(t)=\begin{pmatrix}e^{-t}&e^{5t}\\-e^{-t}&e^{5t}\end{pmatrix}$
and $v=\begin{pmatrix}C₁\\C₂\end{pmatrix}$ is an arbitrary vector.

A matrix $\mathbf{Ψ}(t)$ with this property is called a *fundamental
matrix of solutions* of the system. As we will see below, a system can
have many fundamental matrices of solutions.

A fundamental matrix is useful, e.g., to solve many initial value
problems with the same matrix.

```{prf:example}
Consider the system of linear ODEs given by

$$
x'&=2x+3y\\
y'&=3x+2y
$$

Solve the initial value problems

* $x(0)=1$, $y(0)=-1$;
* $x(0)=-1$, $y(0)=-1$;
* $x(0)=3$, $y(0)=4$.

In each case, we need to find a vector $v$ such that
$\mathbf{Ψ}(0)v=\begin{pmatrix}x(0)\\y(0)\end{pmatrix}$, where $\mathbf{Ψ}(t)=\begin{pmatrix}e^{-t}&e^{5t}\\-e^{-t}&e^{5t}\end{pmatrix}$
is the fundamental matrix we found above.

Compute $\mathbf{Ψ}(0)=\begin{pmatrix}1&1\\-1&1\end{pmatrix}$, hence
we need to solve $C₁+C₂=x(0)$, $-C₁+C₂=y(0)$. Instead of solving this
system in each case, we can solve it once, then substitute the initial
values. We have $C₁=\frac{x(0)-y(0)}{2}$, $C₂=\frac{x(0)+y(0)}{2}$.

This immediately gives us the answers

* $C₁=1$, $C₂=0$, $x=e^{-t}$, $y=-e^{-t}$.
* $C₁=0$, $C₂=-1$, $x=-e^{5t}$, $y=-e^{5t}$.
* $C₁=-0.5$, $C₂=3.5$, $x=-0.5e^{-t}+3.5e^{5t}$,
  $y=0.5e^{-t}+3.5e^{5t}$.
```

If we substitute $C₁=\frac{x(0)-y(0)}{2}$, $C₂=\frac{x(0)+y(0)}{2}$
into the general solution, we get

$$
\begin{pmatrix}x\\y\end{pmatrix}
&=\frac{x(0)-y(0)}{2}e^{-t}\begin{pmatrix}1\\-1\end{pmatrix}+
\frac{x(0)+y(0)}{2}e^{5t}\begin{pmatrix}1\\1\end{pmatrix}\\
&=x(0)\begin{pmatrix}\frac{e^{5t}+e^{-t}}{2}\\\frac{e^{5t}-e^{-t}}{2}\end{pmatrix}
+y(0)\begin{pmatrix}\frac{e^{5t}-e^{-t}}{2}\\\frac{e^{5t}+e^{-t}}{2}\end{pmatrix}\\
&=\begin{pmatrix}\frac{e^{5t}+e^{-t}}{2}&\frac{e^{5t}-e^{-t}}{2}\\\frac{e^{5t}+e^{-t}}{2}&\frac{e^{5t}-e^{-t}}{2}\end{pmatrix}
\begin{pmatrix}x(0)\\y(0)\end{pmatrix}
$$

This means that
$\begin{pmatrix}\frac{e^{5t}+e^{-t}}{2}&\frac{e^{5t}-e^{-t}}{2}\\\frac{e^{5t}+e^{-t}}{2}&\frac{e^{5t}-e^{-t}}{2}\end{pmatrix}$
is another fundamental matrix of the original system.

## Matrix exponential

Again, consider the system

$$
x'&=2x+3y,\\
y'&=3x+2y.
$$

In the previous section, we found two different fundamental matrices
of this sytem,

$$
\mathbf{Ψ₁}&=\begin{pmatrix}e^{-t}&e^{5t}\\-e^{-t}&e^{5t}\end{pmatrix}\\
\mathbf{Ψ₂}&=\begin{pmatrix}\frac{e^{5t}+e^{-t}}{2}&\frac{e^{5t}-e^{-t}}{2}\\\frac{e^{5t}+e^{-t}}{2}&\frac{e^{5t}-e^{-t}}{2}\end{pmatrix}
$$

While the second matrix is more complicated, it has one very important
property: $\mathbf{Ψ₂}(0)=I$, hence the solution of an initial value
problerm is given by
$\mathbf{Ψ₂}(t)\begin{pmatrix}x(0)\\y(0)\end{pmatrix}$. So, this
fundamental matrix allows us to solve initial value problems very
quickly.

It turns out that with a proper definition of the exponential of a
matrix, this fundamental matrix is given by $e^{At}$, thus the
solution of an initial value problem can be written as

$$
\begin{pmatrix}x(t)\\y(t)\end{pmatrix}=
e^{At}\begin{pmatrix}x(0)\\y(0)\end{pmatrix}.
$$

This formula is very similar to the formula $x(t)=x₀e^{λt}$ for the
solution of the one-dimensional initial value problem $x'=λx$,
$x(0)=x₀$. Another important fact about this formula is that computer
algebra systems can find matrix exponentials, thus you can solve an
initial value problem in a few lines of code.

The formal definition of the matrix exponential is provided for
curious readers but will never be used in the course.

```{prf:definition}
If $A$ is a square matrix (i.e., the number of rows is equal to the
number of columns), then $e^A$ is defined as the sum of the series

$$
e^A=\sum_{n=0}^∞ \frac{A^n}{n!},
$$

where $A^n$ means $A$ to the $n$-th power in the sense of matrix
multiplication.
```

To find $e^{At}$ without using a computer, one needs to solve the
system $x'=Ax$ with initial conditions
$x(0)=\begin{pmatrix}1\\0\end{pmatrix}$ and
$x(0)=\begin{pmatrix}0\\1\end{pmatrix}$, then write these solutions as
the first and the second **column** of the answer, respectively.

**NB**: there was a typo in the previous paragraph ("row" instead of
column), fixed on 12/07/2022 in the morning.

```{prf:example}
Find $e^{At}$, where $A=\begin{pmatrix}1&1\\4&1\end{pmatrix}$.

The eigenvalues of $A$ are $-1$ and $3$, the corresponding
eigenvectors are $\begin{pmatrix}1\\-2\end{pmatrix}$ and
$\begin{pmatrix}1\\2\end{pmatrix}$. Hence, $x(t)=C₁e^{-t}+C₂e^{3t}$,
$y(t)=-2C₁e^{-t}+2C₂e^{3t}$ is a general solution of
$\begin{pmatrix}x\\y\end{pmatrix}'=A\begin{pmatrix}x\\y\end{pmatrix}$.

For initial conditions $x(0)=1$, $y(0)=0$, we find $C₁=C₂=1/2$,
$x(t)=\frac{e^{-t}+e^{3t}}{2}$, $y(t)=e^{3t}-e^{-t}$.

For initial conditions $x(0)=0$, $y(0)=1$, we find $C₁=-1/4$,
$C₂=1/4$, $x(t)=\frac{e^{3t}-e^{-t}}{4}$,
$y(t)=\frac{e^{-t}+e^{3t}}{2}$.

Thus

$$
e^{At}=
  \begin{pmatrix}
	\frac{e^{-t}+e^{3t}}{2}&\frac{e^{3t}-e^{-t}}{4}\\
	e^{3t}-e^{-t}&\frac{e^{-t}+e^{3t}}{2}
  \end{pmatrix}
$$
```

## Solving systems of linear ODEs with computer

TODO

## Method of undetermined coefficients

In this section we discuss how to solve a linear system
$x'(t)=Ax(t)+b(t)$, where the coordinates of $b(t)$ are linear
combinations of the terms $t^ke^{λt}$, $t^ke^{λt}\cos(μt)$,
$t^ke^{λt}\sin(μt)$.

As in the case of higher order linear ODEs, we find the complementary
function $x_c$ (a.k.a. the general solution of $x_c'=Ax_c$), then find
a particular solution $x_p$, and write the answer $x=x_c+x_p$. To find
a “good guess” that is guaranteed to work, we use almost the same
rules as the ones we used for higher order ODEs[^rules], with one
important modification: if a term (e.g., $e^{λt}$) appears in the
complementary function, then we need **both** $A₁e^{λt}$ and
$A₂te^{λt}$ in the “guess”; if $λ$ is an eigenvalue of multiplicity
$2$, then we also need $A₃t²e^{λt}$.

[^rules]: TODO: repeat the rules here

For examples, see [scans from lecture on Nov 28](./_static/Nov28.pdf)
and [photos of the whiteboard from Nov 30](./_static/20221130.zip)
