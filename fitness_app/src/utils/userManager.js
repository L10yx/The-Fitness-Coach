/**
 * 用户账号管理工具类
 * 用于处理用户登录状态和数据存储
 */

// 用户信息存储键名
const USER_INFO_KEY = 'fitness_user_info';
const USER_EXERCISE_HISTORY_KEY = 'fitness_exercise_history';
const USER_PROFILE_KEY = 'fitness_user_profile';

/**
 * 用户管理类
 */
export default {
  /**
   * 获取当前登录用户信息
   * @returns {Object|null} 用户信息对象，未登录则返回null
   */
  getCurrentUser() {
    try {
      const userInfoStr = uni.getStorageSync(USER_INFO_KEY);
      return userInfoStr ? JSON.parse(userInfoStr) : null;
    } catch (e) {
      console.error('获取用户信息失败', e);
      return null;
    }
  },

  /**
   * 保存用户信息到本地存储
   * @param {Object} userInfo 用户信息对象
   */
  saveUserInfo(userInfo) {
    if (!userInfo) return;
    try {
      uni.setStorageSync(USER_INFO_KEY, JSON.stringify(userInfo));
    } catch (e) {
      console.error('保存用户信息失败', e);
    }
  },

  /**
   * 用户登录
   * @param {String} username 用户名
   * @param {String} password 密码
   * @returns {Promise} 登录结果
   */
  login(username, password) {
    // 这里应该是实际的登录API调用
    // 这里仅作为示例，模拟登录成功
    return new Promise((resolve) => {
      // 模拟网络请求延迟
      setTimeout(() => {
        // 创建模拟用户数据
        const userInfo = {
          id: '10086',
          username: username,
          nickname: '健身达人',
          level: '初级会员',
          avatar: '',
          createdAt: new Date().toISOString()
        };
        
        // 保存到本地存储
        this.saveUserInfo(userInfo);
        
        resolve({
          success: true,
          data: userInfo
        });
      }, 500);
    });
  },

  /**
   * 用户退出登录
   */
  logout() {
    try {
      uni.removeStorageSync(USER_INFO_KEY);
      return true;
    } catch (e) {
      console.error('退出登录失败', e);
      return false;
    }
  },

  /**
   * 获取用户运动历史记录
   * @returns {Array} 历史记录数组
   */
  getUserExerciseHistory() {
    const user = this.getCurrentUser();
    if (!user) return [];
    
    try {
      // 获取用户特定的历史记录
      const key = `${USER_EXERCISE_HISTORY_KEY}_${user.id}`;
      const historyStr = uni.getStorageSync(key);
      return historyStr ? JSON.parse(historyStr) : [];
    } catch (e) {
      console.error('获取运动历史记录失败', e);
      return [];
    }
  },

  /**
   * 保存用户运动历史记录
   * @param {Array} records 历史记录数组
   */
  saveUserExerciseHistory(records) {
    const user = this.getCurrentUser();
    if (!user) return false;
    
    try {
      // 保存到用户特定的存储键下
      const key = `${USER_EXERCISE_HISTORY_KEY}_${user.id}`;
      uni.setStorageSync(key, JSON.stringify(records));
      return true;
    } catch (e) {
      console.error('保存运动历史记录失败', e);
      return false;
    }
  },

  /**
   * 添加一条运动记录
   * @param {Object} record 运动记录对象
   */
  addExerciseRecord(record) {
    if (!record) return false;
    
    // 获取现有记录
    const records = this.getUserExerciseHistory();
    
    // 添加新记录到开头
    records.unshift({
      ...record,
      id: Date.now().toString(), // 生成唯一ID
      createdAt: new Date().toISOString()
    });
    
    // 保存更新后的记录
    return this.saveUserExerciseHistory(records);
  },

  /**
   * 删除一条运动记录
   * @param {String} recordId 记录ID
   */
  deleteExerciseRecord(recordId) {
    if (!recordId) return false;
    
    // 获取现有记录
    const records = this.getUserExerciseHistory();
    
    // 找到并删除指定记录
    const index = records.findIndex(item => item.id === recordId);
    if (index !== -1) {
      records.splice(index, 1);
      // 保存更新后的记录
      return this.saveUserExerciseHistory(records);
    }
    
    return false;
  },

  /**
   * 获取用户运动统计数据
   * @returns {Object} 统计数据对象
   */
  /**
   * 获取用户个人资料
   * @returns {Object|null} 用户个人资料对象，未设置则返回null
   */
  getUserProfile() {
    const user = this.getCurrentUser();
    if (!user) return null;
    
    try {
      // 获取用户特定的个人资料
      const key = `${USER_PROFILE_KEY}_${user.id}`;
      const profileStr = uni.getStorageSync(key);
      return profileStr ? JSON.parse(profileStr) : null;
    } catch (e) {
      console.error('获取用户个人资料失败', e);
      return null;
    }
  },
  
  /**
   * 保存用户个人资料
   * @param {Object} profile 用户个人资料对象
   * @returns {Boolean} 保存结果
   */
  saveUserProfile(profile) {
    const user = this.getCurrentUser();
    if (!user) return false;
    
    try {
      // 保存到用户特定的存储键下
      const key = `${USER_PROFILE_KEY}_${user.id}`;
      uni.setStorageSync(key, JSON.stringify(profile));
      return true;
    } catch (e) {
      console.error('保存用户个人资料失败', e);
      return false;
    }
  },
  
  getUserExerciseStats() {
    const records = this.getUserExerciseHistory();
    
    // 默认统计数据
    const stats = {
      totalDays: 0,
      consecutiveDays: 0,
      totalCount: 0,
      exerciseTypes: {}
    };
    
    if (records.length === 0) return stats;
    
    // 计算不同日期的训练天数
    const trainingDays = new Set();
    
    // 计算总次数和各类型运动次数
    records.forEach(record => {
      // 提取日期部分
      const dateStr = record.date.split('T')[0] || record.date.split(' ')[0];
      trainingDays.add(dateStr);
      
      // 累加总次数
      stats.totalCount += record.count || 0;
      
      // 按类型统计
      const type = record.type;
      if (!stats.exerciseTypes[type]) {
        stats.exerciseTypes[type] = 0;
      }
      stats.exerciseTypes[type] += record.count || 0;
    });
    
    // 设置总训练天数
    stats.totalDays = trainingDays.size;
    
    // 计算连续训练天数
    const sortedDays = Array.from(trainingDays).sort();
    let currentStreak = 1;
    let maxStreak = 1;
    
    // 检查今天是否有训练
    const today = new Date().toISOString().split('T')[0];
    const hasTrainedToday = sortedDays.includes(today);
    
    // 如果有多天记录，计算最大连续天数
    if (sortedDays.length > 1) {
      for (let i = 1; i < sortedDays.length; i++) {
        const prevDate = new Date(sortedDays[i-1]);
        const currDate = new Date(sortedDays[i]);
        
        // 检查日期是否连续
        const diffDays = Math.round((currDate - prevDate) / (24 * 60 * 60 * 1000));
        
        if (diffDays === 1) {
          // 日期连续
          currentStreak++;
          maxStreak = Math.max(maxStreak, currentStreak);
        } else {
          // 日期不连续，重置计数
          currentStreak = 1;
        }
      }
    }
    
    // 如果今天没有训练，检查最后一天是否是昨天
    if (!hasTrainedToday && sortedDays.length > 0) {
      const lastTrainingDay = new Date(sortedDays[sortedDays.length - 1]);
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      yesterday.setHours(0, 0, 0, 0);
      
      // 如果最后训练日不是昨天，重置连续天数
      if (lastTrainingDay.getTime() !== yesterday.getTime()) {
        currentStreak = 0;
      }
    }
    
    stats.consecutiveDays = currentStreak;
    
    return stats;
  }
};