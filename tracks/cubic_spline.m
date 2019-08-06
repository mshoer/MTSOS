
x = [9:0.1:9.9, 10*ones(1,11)];
xx = [9:0.01:10,10*ones(1,100)];
y = [-5*ones(1,10), -5:0.1:-4];
yy = [-5*ones(1,100), -5:0.01:-4];
theta = [9:0.1:11];

% sx = spline(theta, x);
% sy = spline(theta, y);
ss = spline(x, y);
figure; hold on;
plot(x, y, 'o', xx, yy, '-b', 'linewidth', 2)
plot(xx, ppval(yy))
% plot(ppval(sx,xx), ppval(sy,yy), '-or', 'linewidth', 2)
% axis([-11, 11, -6, 6])