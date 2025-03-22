<template>
  <view class="profile-container">
    <!-- 顶部导航栏 -->
    <view class="nav-bar">
      <view class="back-button" @click="goBack">
        <text class="back-icon">←</text>
      </view>
      <text class="page-title">个人中心</text>
      <view class="placeholder"></view>
    </view>
    
    <!-- 用户信息卡片 -->
    <view class="user-card">
      <view class="avatar-large">
        <text class="avatar-text">{{ getAvatarText() }}</text>
      </view>
      <view class="user-info">
        <text class="user-name">{{ userData.name }}</text>
        <text class="user-id">ID: {{ userData.id }}</text>
      </view>
      <view class="edit-profile" @click="editProfile">
        <text class="edit-text">编辑</text>
      </view>
    </view>
    
    <!-- 个人数据统计 -->
    <view class="stats-container">
      <view class="stat-item">
        <text class="stat-value">{{ exerciseStats.totalDays }}</text>
        <text class="stat-label">训练天数</text>
      </view>
      <view class="stat-item">
        <text class="stat-value">{{ exerciseStats.consecutiveDays }}</text>
        <text class="stat-label">连续天数</text>
      </view>
      <view class="stat-item">
        <text class="stat-value">{{ exerciseStats.totalCount }}</text>
        <text class="stat-label">总计数</text>
      </view>
    </view>
    
    <!-- 功能列表 -->
    <view class="section-title">个人资料</view>
    <view class="menu-list">
      <view class="menu-item" @click="goToPersonalInfo">
        <text class="menu-icon">👤</text>
        <text class="menu-text">基本信息</text>
        <text class="menu-arrow">→</text>
      </view>
      <view class="menu-item" @click="goToHealthInfo">
        <text class="menu-icon">❤️</text>
        <text class="menu-text">健康数据</text>
        <text class="menu-arrow">→</text>
      </view>
      <view class="menu-item" @click="goToAchievements">
        <text class="menu-icon">🏆</text>
        <text class="menu-text">我的成就</text>
        <text class="menu-arrow">→</text>
      </view>
    </view>
    
    <view class="section-title">账号管理</view>
    <view class="menu-list">
      <view class="menu-item" @click="goToSettings">
        <text class="menu-icon">⚙️</text>
        <text class="menu-text">设置</text>
        <text class="menu-arrow">→</text>
      </view>
      <view class="menu-item" @click="goToHelp">
        <text class="menu-icon">❓</text>
        <text class="menu-text">帮助与反馈</text>
        <text class="menu-arrow">→</text>
      </view>
      <view class="menu-item" @click="goToAbout">
        <text class="menu-icon">ℹ️</text>
        <text class="menu-text">关于我们</text>
        <text class="menu-arrow">→</text>
      </view>
      <view class="menu-item logout" @click="logout">
        <text class="menu-icon">🚪</text>
        <text class="menu-text">退出登录</text>
        <text class="menu-arrow">→</text>
      </view>
    </view>
    
    <!-- 版本信息 -->
    <view class="version-info">
      <text class="version-text">SFC v1.0.0</text>
    </view>
  </view>
</template>

<script>
import userManager from '@/utils/userManager.js';

export default {
  data() {
    return {
      // 用户数据
      userData: {
        name: '健身达人',
        id: '10086'
      },
      // 运动统计数据
      exerciseStats: {
        totalDays: 0,
        consecutiveDays: 0,
        totalCount: 0
      }
    }
  },
  onShow() {
    // 每次页面显示时更新用户信息和统计数据
    this.updateUserInfo();
    this.updateExerciseStats();
  },
  
  methods: {
    goBack() {
      uni.navigateBack();
    },
    editProfile() {
      uni.navigateTo({
        url: '/pages/profile/profile_edit'
      });
    },
    goToPersonalInfo() {
      uni.showToast({
        title: '基本信息功能开发中',
        icon: 'none'
      });
    },
    goToHealthInfo() {
      uni.showToast({
        title: '健康数据功能开发中',
        icon: 'none'
      });
    },
    goToAchievements() {
      uni.showToast({
        title: '成就功能开发中',
        icon: 'none'
      });
    },
    goToSettings() {
      uni.navigateTo({
        url: '/pages/settings/settings'
      });
    },
    goToHelp() {
      uni.showToast({
        title: '帮助与反馈功能开发中',
        icon: 'none'
      });
    },
    goToAbout() {
      uni.showToast({
        title: '关于我们功能开发中',
        icon: 'none'
      });
    },
    updateUserInfo() {
      // 获取当前登录用户信息
      const userInfo = userManager.getCurrentUser();
      if (userInfo) {
        this.userData = {
          name: userInfo.nickname || userInfo.username,
          id: userInfo.id
        };
      }
    },
    
    // 获取头像文本
    getAvatarText() {
      if (this.userData && this.userData.name) {
        return this.userData.name.charAt(0);
      }
      return '用户';
    },
    
    updateExerciseStats() {
      // 获取用户运动统计数据
      const stats = userManager.getUserExerciseStats();
      this.exerciseStats = {
        totalDays: stats.totalDays,
        consecutiveDays: stats.consecutiveDays,
        totalCount: stats.totalCount
      };
    },
    
    logout() {
      uni.showModal({
        title: '提示',
        content: '确定要退出登录吗？',
        success: (res) => {
          if (res.confirm) {
            // 使用userManager退出登录
            userManager.logout();
            
            uni.showToast({
              title: '已退出登录',
              icon: 'success',
              duration: 1500
            });
            
            // 退出后返回到index页面
            setTimeout(() => {
              uni.reLaunch({
                url: '/pages/index/index'
              });
            }, 1500);
          }
        }
      });
    }
  }
}
</script>

<style>
.profile-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: #121212;
  color: #ffffff;
  padding-bottom: 40rpx;
  box-sizing: border-box;
}

/* 导航栏样式 */
.nav-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 90rpx;
  padding: 0 30rpx;
  margin-top: 40rpx;
}

.back-button {
  width: 60rpx;
  height: 60rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 30rpx;
}

.back-icon {
  font-size: 36rpx;
  color: #ffffff;
}

.page-title {
  font-size: 36rpx;
  font-weight: bold;
  color: #ffffff;
}

.placeholder {
  width: 60rpx;
}

/* 用户卡片样式 */
.user-card {
  display: flex;
  align-items: center;
  margin: 40rpx 30rpx;
  padding: 30rpx;
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: 20rpx;
}

.avatar-large {
  width: 120rpx;
  height: 120rpx;
  border-radius: 60rpx;
  background: linear-gradient(135deg, #FF4785 0%, #FF8D4E 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 30rpx;
}

.avatar-text {
  font-size: 40rpx;
  color: #ffffff;
  font-weight: bold;
}

.user-info {
  flex: 1;
}

.user-name {
  font-size: 36rpx;
  font-weight: bold;
  color: #ffffff;
  margin-bottom: 10rpx;
}

.user-id {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
}

.edit-profile {
  width: 100rpx;
  height: 60rpx;
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 30rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}

.edit-text {
  font-size: 26rpx;
  color: #ffffff;
}

/* 统计数据样式 */
.stats-container {
  display: flex;
  justify-content: space-around;
  margin: 0 30rpx 40rpx;
  padding: 20rpx 0;
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: 20rpx;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stat-value {
  font-size: 40rpx;
  font-weight: bold;
  color: #FF4785;
  margin-bottom: 10rpx;
}

.stat-label {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
}

/* 菜单列表样式 */
.section-title {
  font-size: 32rpx;
  font-weight: bold;
  color: #ffffff;
  margin: 30rpx 30rpx 20rpx;
}

.menu-list {
  margin: 0 30rpx 30rpx;
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: 20rpx;
  overflow: hidden;
}

.menu-item {
  display: flex;
  align-items: center;
  padding: 30rpx;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.menu-item:last-child {
  border-bottom: none;
}

.menu-icon {
  font-size: 36rpx;
  margin-right: 20rpx;
}

.menu-text {
  flex: 1;
  font-size: 30rpx;
  color: #ffffff;
}

.menu-arrow {
  font-size: 30rpx;
  color: rgba(255, 255, 255, 0.6);
}

.logout .menu-text {
  color: #FF4785;
}

/* 版本信息样式 */
.version-info {
  display: flex;
  justify-content: center;
  margin-top: auto;
  padding: 30rpx 0;
}

.version-text {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.4);
}
</style>