# 1. 初始化（会自动处理连接池）
from redisClient import RedisClient


db = RedisClient(host='127.0.0.1', port=6379)

# 2. 增/改
db.set('user:1001', 'Gemini', ex=3600)  # 设置 1 小时过期

# 3. 查
name = db.get('user:1001')
print(f"获取到的用户名: {name}")

# 4. 删
db.delete('user:1001')

# 5. 判断是否存在
if not db.exists('user:1001'):
    print("该用户缓存已清理")