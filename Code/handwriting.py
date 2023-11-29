import sys
import tensorflow as tf
import matplotlib.pyplot as plt

# 使用MNIST训练集
mnist = tf.keras.datasets.mnist

# 用于训练的数据设置
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# 将分类标签映射为向量
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
# 根据模型的输入要求，将三维数组改变为四维数组
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)

# 创建前馈卷积神经网络
model = tf.keras.models.Sequential([

    # 卷积层，使用32个3x3卷积核
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(28, 28, 1)
    ),

    # 使用最大值池化
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # 将多维数组展平为一维数组
    tf.keras.layers.Flatten(),

    # 添加隐藏层
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # 输出层
    tf.keras.layers.Dense(10, activation="softmax")
])

# 配置神经网络模型
model.compile(
    optimizer="adam",   # 配置优化算法
    metrics=["accuracy"],    # 监控检测指标
    # 配置损失函数
    loss="categorical_crossentropy"    # 交叉熵损失函数
    # loss="mean_squared_error"     # 均方误差损失函数
    # loss="mean_absolute_error"    # 平均绝对误差损失函数
)


# 定义自定义回调函数
class AccuracyHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('accuracy'))


accuracy_history = AccuracyHistory()
# 进行训练,训练的轮数为10
history = model.fit(x_train, y_train, epochs=10, callbacks=[accuracy_history])

# 模型评估
model.evaluate(x_test,  y_test, verbose=2)

# 将模型保存在文件中
if len(sys.argv) == 2:
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}.")


# 保存loss变化图像
train_loss = history.history['loss']
plt.plot(range(1, 11), train_loss, label='Training Loss')
plt.legend()
plt.ylabel("Loss")
plt.xlabel('Epochs')
plt.title("loss function: categorical_crossentropy")
plt.savefig("../Result/categorical_crossentropy/loss.png")
plt.clf()

# 保存准确率变化图像
train_accuracy = history.history['accuracy']
plt.plot(range(1, 11), train_accuracy, label='Training Accuracy')
plt.legend()
plt.ylabel("Accuracy")
plt.xlabel('Epochs')
plt.title("loss function: categorical_crossentropy")
plt.savefig("../Result/categorical_crossentropy/accuracy.png")
plt.show()
