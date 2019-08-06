clear all
close all

%% Discretization Parameters
stepsize = 0.05;
maxsval = 0.95;

%% Bezier Control Points, Bernstein Basis
P = [ 0, -5;
      1, -5;
      2, -5;
      3, -5;
      4, -5;
      5, -4;
      5, -3;
      5, -2;
      5, -1;
      5,  0];

% P = [ 0, -5;
%     1, -5;
%     2, -5;
%     3, -5];
  
%% Compute Path
x = [];
y = [];
syms t
B = bernsteinMatrix(size(P,1)-1,t);
bezierCurve = simplify(B*P);
fplot(bezierCurve(1), bezierCurve(2), [0, 1.0])
hold on
scatter(P(:,1), P(:,2),'filled', 'blue')
plot(P(:,1),P(:,2), '--', 'color', 'black');

x = [x double(subs(bezierCurve(1),0:stepsize:maxsval))];
y = [y double(subs(bezierCurve(2),0:stepsize:maxsval))];

title('rectangular')

% csvwrite('rectangular.txt', [x; y])