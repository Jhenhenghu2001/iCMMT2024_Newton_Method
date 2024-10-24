% 定義網格範圍
x = linspace(0, 1000, 200); % X 軸範圍
y = linspace(0, 1000, 200); % Y 軸範圍
[X, Y] = meshgrid(x, y);

% 定義矩形範圍
rect_x_min = 350;  % 矩形X的最小值
rect_x_max = 650;  % 矩形X的最大值
rect_y_min = 350;  % 矩形Y的最小值
rect_y_max = 650;  % 矩形Y的最大值

% 初始化危險因子矩陣
danger_factor = zeros(size(X));

% 矩形內的危險因子設為1
inside_rect = (X >= rect_x_min & X <= rect_x_max & Y >= rect_y_min & Y <= rect_y_max);
danger_factor(inside_rect) = 1;

% 定義矩形外的危險因子隨距離遞減
for i = 1:length(x)
    for j = 1:length(y)
        if ~inside_rect(i,j)
            % 計算到最近矩形邊界的距離
            dx = max([rect_x_min - X(i,j), 0, X(i,j) - rect_x_max]);
            dy = max([rect_y_min - Y(i,j), 0, Y(i,j) - rect_y_max]);
            dist_to_rect = sqrt(dx^2 + dy^2);
            
            % 危險因子根據距離遞減
            danger_factor(i,j) = max(1 - dist_to_rect/50, 0); % 50 為調整危險因子衰減的參數
        end
    end
end

% 繪製立體圖
figure;
mesh(X, Y, danger_factor);
xlabel('X');
ylabel('Y');
zlabel('Danger Factor');
% title('危險因子三維即視圖');
colorbar;  % 顯示顏色條
colormap(jet);
caxis([0 1]);
view(3);   % 設置三維視角