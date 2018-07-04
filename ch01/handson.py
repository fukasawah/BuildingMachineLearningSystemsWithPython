import scipy as sp
data = sp.genfromtxt("data/web_traffic.tsv", delimiter="\t")

# 先頭10件を表示
print(data[:10])

# 1列目を1次元配列で取り出す(経過時間？)
x = data[:, 0]

# 2列目を1次元配列で取り出す(アクセス数)
y = data[:, 1]

# 不正なデータが入っている要素を除去(アクセス数が不正な値(NAN)の行を除去)

invalid_data = ~sp.isnan(y)
x = x[invalid_data]
y = y[invalid_data]

print("x[-1] = %d" % x[-1])

# フォント
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAexGothic'

# 近似関数(1次)を得る
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
print("Model parameters: %s" % fp1)
print("residuals: %s" % residuals)

# 近似関数からモデル関数を作る
f1 = sp.poly1d(fp1)

# モデル関数(f1)と実測値(x,y)から誤差を得る
def error(f,x,y):
    return sp.sum((f(x) - y) ** 2)

print(error(f1, x, y )) # 317389767.339778


# 近似関数(2次)を得る
fp2, residuals, rank, sv, rcond = sp.polyfit(x, y, 2, full=True)
print("Model parameters: %s" % fp2) # Model parameters: [ 1.05322215e-02 -5.26545650e+00  1.97476082e+03]
print("residuals: %s" % residuals) # residuals: [1.79983508e+08]

# 近似関数からモデル関数を作る
f2 = sp.poly1d(fp2)
print(error(f2, x, y )) # 179983507.8781792 (f1と比べて半分ぐらいに減った！)


# 表示
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.title("先月のWebトラフィック量")
plt.xlabel("時間")
plt.ylabel("件/時")
plt.xticks([w*7*24 for w in range(10)], ['week %i'% w for w in range(10)]) # 1週間(7日*24時間)毎に区切る。0周目～9周目まで。
plt.autoscale(tight=True) # 実際は4周目分のデータしかないので、表示を0周目～4周目までを収まるようにスケール
plt.grid()

# # モデル関数f1を使い、線を引く
fx = sp.linspace(0, x[-1], 1000) # 間隔が等しい数列を1000個作る。始点:0,終点:x[-1]=743,数:1000個 => 約0.743づつ増加する1000個の数列
# # print(sp.linspace(0, x[-1], 1000)) # 数列を確認

# 変化がある3.5週目から分けて２つの直線で近似させる
inflection = int(3.5 * 7 * 24)
xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)
print("fa_error = %s" % fa_error )
print("fb_error = %s" % fb_error )
print("fa_error+fb_error = %s" % (fa_error + fb_error) )


# 3.5週目以降のデータのうち、
# 30%をテストデータとして使う

# 再現性を出すためのseed固定
sp.random.seed(314159263)

frac = 0.30
split_idx = int(frac * len(xb))
shuffled = sp.random.permutation(list(range(len(xb))))
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])

fbt1 = sp.poly1d(sp.polyfit(xb[train], yb[train], 1))
fbt2 = sp.poly1d(sp.polyfit(xb[train], yb[train], 2))
fbt3 = sp.poly1d(sp.polyfit(xb[train], yb[train], 3))
fbt10 = sp.poly1d(sp.polyfit(xb[train], yb[train], 10))
fbt100 = sp.poly1d(sp.polyfit(xb[train], yb[train], 100))

# 学習データで誤差を見る
print("(train)fbt1_error  = %s" % error(fbt1, xb[train], yb[train]) ) # 15846249.623390418
print("(train)fbt2_error  = %s" % error(fbt2, xb[train], yb[train]) ) # 14048047.808891317
print("(train)fbt3_error  = %s" % error(fbt3, xb[train], yb[train]) ) # 14014812.664947439
print("(train)fbt10_error = %s" % error(fbt10, xb[train], yb[train]) )# 13092597.979928792
print("(train)fbt100_error = %s" % error(fbt100, xb[train], yb[train]) )# 12652070.062779315 <= 最も誤差が小さい、が、学習データに最適化されている状態なので、評価としては不適切。

# 未知データで誤差を見る
print("(test) fbt1_error  = %s" % error(fbt1, xb[test], yb[test]) ) # 6387914.410987866
print("(test) fbt2_error  = %s" % error(fbt2, xb[test], yb[test]) ) # 5841997.167708391 <= 最も誤差が小さい。乱数を変えても大体これが良い結果に。
print("(test) fbt3_error  = %s" % error(fbt3, xb[test], yb[test]) ) # 5949556.660118614
print("(test) fbt10_error = %s" % error(fbt10, xb[test], yb[test]) )# 6525801.591164470
print("(test)fbt100_error = %s" % error(fbt100, xb[test], yb[test]))# 6677906.164012078


### 描画 ###


# 未来を含めて描画(10週目まで描画してみる。)
fx = sp.linspace(0, 7*24*10, 1000)

# # f1
# plt.plot(fx, f1(fx), linewidth=2) # 直線を描く(xに対するy(f1(x))の線を引く)
# plt.legend(["d=%i" % f1.order], loc="upper left") # 凡例

# # f2
# plt.plot(fx, f2(fx), linewidth=2) # 直線を描く(xに対するy(f1(x))の線を引く)
# plt.legend(["d=%i" % f2.order], loc="upper left") # 凡例

# # 変化がある3.5週目から分けて２つの直線を引く(fa,fb)
# plt.plot(fx, fa(fx), linewidth=2) 
# plt.plot(fx, fb(fx), linewidth=2) 
# plt.legend(["d=%i" % fa.order, "d=%i" % fb.order], loc="upper left") # 凡例

# 変化がある3.5週目から70%を学習に使って作ったモデルで線を引く
plt.plot(fx, fbt1(fx), linewidth=2) 
plt.plot(fx, fbt2(fx), linewidth=2) 
plt.plot(fx, fbt3(fx), linewidth=2) 
plt.plot(fx, fbt10(fx), linewidth=2) 
plt.plot(fx, fbt100(fx), linewidth=2) 

plt.legend(["d=%i" % fbt1.order, "d=%i" % fbt2.order, "d=%i" % fbt3.order, "d=%i" % fbt10.order, "d=%i" % fbt100.order], loc="upper left") # 凡例


plt.show()

