import numpy as np
import pygame
import sys
import tensorflow as tf
import time

# 加载模型
if len(sys.argv) != 2:
    sys.exit("Usage: python recognition.py model")
model = tf.keras.models.load_model(sys.argv[1])

# 颜色初始化
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# 初始化pygame模块
pygame.init()
size = width, height = 600, 400
screen = pygame.display.set_mode(size)

# 加载字体
OPEN_SANS = "OpenSans-Regular.ttf"
smallFont = pygame.font.Font(OPEN_SANS, 20)
largeFont = pygame.font.Font(OPEN_SANS, 40)

# 设置画板行列数
ROWS, COLS = 28, 28

OFFSET = 20
CELL_SIZE = 10

# 建立二维数组做画板
handwriting = [[0] * COLS for _ in range(ROWS)]
classification = None

# 运行pygame
while True:

    # 检查是否退出
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill(BLACK)

    # 监听鼠标，并得到鼠标点击位置
    click, _, _ = pygame.mouse.get_pressed()
    if click == 1:
        mouse = pygame.mouse.get_pos()
    else:
        mouse = None

    # 绘制网格
    cells = []
    for i in range(ROWS):
        row = []
        for j in range(COLS):
            rect = pygame.Rect(
                OFFSET + j * CELL_SIZE,
                OFFSET + i * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )

            # 如果该网格被点击
            if handwriting[i][j]:
                channel = 255 - (handwriting[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)

            else:
                pygame.draw.rect(screen, WHITE, rect)
            #     绘制格子框
            pygame.draw.rect(screen, BLACK, rect, 1)

            # 填充被点击网格的邻居格子
            if mouse and rect.collidepoint(mouse):
                handwriting[i][j] = 250 / 255
                if i + 1 < ROWS:
                    handwriting[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    handwriting[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    handwriting[i + 1][j + 1] = 190 / 255

    # Reset按钮
    resetButton = pygame.Rect(
        30, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    resetText = smallFont.render("Reset", True, BLACK)
    resetTextRect = resetText.get_rect()
    resetTextRect.center = resetButton.center
    pygame.draw.rect(screen, WHITE, resetButton)
    screen.blit(resetText, resetTextRect)

    # Classify按钮
    classifyButton = pygame.Rect(
        150, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    classifyText = smallFont.render("Classify", True, BLACK)
    classifyTextRect = classifyText.get_rect()
    classifyTextRect.center = classifyButton.center
    pygame.draw.rect(screen, WHITE, classifyButton)
    screen.blit(classifyText, classifyTextRect)

    # 重置图像
    if mouse and resetButton.collidepoint(mouse):
        handwriting = [[0] * COLS for _ in range(ROWS)]
        classification = None

    # 模型识别
    if mouse and classifyButton.collidepoint(mouse):
        classification = model.predict(
            [np.array(handwriting).reshape(1, 28, 28, 1)]
        ).argmax()

    # 显示模型识别结果
    if classification is not None:
        classificationText = largeFont.render(str(classification), True, WHITE)
        classificationRect = classificationText.get_rect()
        grid_size = OFFSET * 2 + CELL_SIZE * COLS
        classificationRect.center = (
            grid_size + ((width - grid_size) / 2),
            100
        )
        screen.blit(classificationText, classificationRect)

    pygame.display.flip()
