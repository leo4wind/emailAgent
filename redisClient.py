import redis

class RedisClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局只有一个连接池"""
        if not cls._instance:
            cls._instance = super(RedisClient, cls).__new__(cls)
        return cls._instance

    def __init__(self, host='localhost', port=6379, db=0, password=None):
        # 建立连接池
        self.pool = redis.ConnectionPool(
            host=host, 
            port=port, 
            db=db, 
            password=password, 
            decode_responses=True  # 关键：自动将字节码转为字符串
        )
        self.client = redis.StrictRedis(connection_pool=self.pool)

    # --- 增 / 改 (Create / Update) ---
    def set(self, key, value, ex=None):
        """设置 key-value，ex 为过期时间（秒）"""
        return self.client.set(key, value, ex=ex)

    # --- 查 (Read) ---
    def get(self, key):
        """获取指定 key 的值"""
        return self.client.get(key)

    def exists(self, key):
        """判断 key 是否存在"""
        return self.client.exists(key)

    # --- 删 (Delete) ---
    def delete(self, *keys):
        """删除一个或多个 key"""
        return self.client.delete(*keys)

    # --- 高级操作：Hash 增删改查 ---
    def hset(self, name, key, value):
        """哈希表设置字段"""
        return self.client.hset(name, key, value)

    def hget(self, name, key):
        """哈希表获取字段"""
        return self.client.hget(name, key)