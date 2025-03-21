<template>
  <view class="settings-container">
    <!-- 顶部导航栏 -->
    <view class="nav-bar">
      <view class="back-button" @click="goBack">
        <text class="back-icon">←</text>
      </view>
      <text class="page-title">设置</text>
      <view class="placeholder"></view>
    </view>
    
    <!-- 设置列表 -->
    <view class="section-title">权限管理</view>
    <view class="settings-list">
      <view class="settings-item">
        <view class="settings-left">
          <text class="settings-icon">📷</text>
          <text class="settings-text">相机权限</text>
        </view>
        <switch :checked="permissions.camera" @change="togglePermission('camera')" color="#FF4785" />
      </view>
      <view class="settings-item">
        <view class="settings-left">
          <text class="settings-icon">🔊</text>
          <text class="settings-text">麦克风权限</text>
        </view>
        <switch :checked="permissions.microphone" @change="togglePermission('microphone')" color="#FF4785" />
      </view>
      <view class="settings-item">
        <view class="settings-left">
          <text class="settings-icon">📱</text>
          <text class="settings-text">通知权限</text>
        </view>
        <switch :checked="permissions.notification" @change="togglePermission('notification')" color="#FF4785" />
      </view>
      <view class="settings-item">
        <view class="settings-left">
          <text class="settings-icon">📍</text>
          <text class="settings-text">位置权限</text>
        </view>
        <switch :checked="permissions.location" @change="togglePermission('location')" color="#FF4785" />
      </view>
    </view>
    
    <view class="section-title">通知设置</view>
    <view class="settings-list">
      <view class="settings-item">
        <view class="settings-left">
          <text class="settings-icon">🔔</text>
          <text class="settings-text">训练提醒</text>
        </view>
        <switch :checked="notifications.training" @change="toggleNotification('training')" color="#FF4785" />
      </view>
      <view class="settings-item">
        <view class="settings-left">
          <text class="settings-icon">🏆</text>
          <text class="settings-text">成就通知</text>
        </view>
        <switch :checked="notifications.achievement" @change="toggleNotification('achievement')" color="#FF4785" />
      </view>
      <view class="settings-item">
        <view class="settings-left">
          <text class="settings-icon">📊</text>
          <text class="settings-text">数据报告</text>
        </view>
        <switch :checked="notifications.report" @change="toggleNotification('report')" color="#FF4785" />
      </view>
    </view>
    
    <view class="section-title">隐私与安全</view>
    <view class="settings-list">
      <view class="settings-item" @click="goToPrivacyPolicy">
        <view class="settings-left">
          <text class="settings-icon">🔒</text>
          <text class="settings-text">隐私政策</text>
        </view>
        <text class="settings-arrow">→</text>
      </view>
      <view class="settings-item" @click="goToTerms">
        <view class="settings-left">
          <text class="settings-icon">📜</text>
          <text class="settings-text">用户协议</text>
        </view>
        <text class="settings-arrow">→</text>
      </view>
      <view class="settings-item" @click="clearCache">
        <view class="settings-left">
          <text class="settings-icon">🗑️</text>
          <text class="settings-text">清除缓存</text>
        </view>
        <text class="settings-arrow">→</text>
      </view>
    </view>
    
    <view class="section-title">其他设置</view>
    <view class="settings-list">
      <view class="settings-item">
        <view class="settings-left">
          <text class="settings-icon">🌙</text>
          <text class="settings-text">深色模式</text>
        </view>
        <switch :checked="otherSettings.darkMode" @change="toggleOtherSetting('darkMode')" color="#FF4785" />
      </view>
      <view class="settings-item" @click="checkUpdate">
        <view class="settings-left">
          <text class="settings-icon">🔄</text>
          <text class="settings-text">检查更新</text>
        </view>
        <text class="settings-arrow">→</text>
      </view>
      <view class="settings-item" @click="goToFeedback">
        <view class="settings-left">
          <text class="settings-icon">💬</text>
          <text class="settings-text">意见反馈</text>
        </view>
        <text class="settings-arrow">→</text>
      </view>
    </view>
  </view>
</template>

<script>
export default {
  data() {
    return {
      permissions: {
        camera: false,
        microphone: false,
        notification: true,
        location: false
      },
      notifications: {
        training: true,
        achievement: true,
        report: false
      },
      otherSettings: {
        darkMode: true
      }
    }
  },
  methods: {
    goBack() {
      uni.navigateBack();
    },
    togglePermission(type) {
      this.permissions[type] = !this.permissions[type];
      // 这里可以添加实际的权限请求逻辑
      console.log(`${type}权限状态:`, this.permissions[type]);
      
      // 模拟权限请求
      if (this.permissions[type]) {
        uni.showToast({
          title: `已开启${type}权限`,
          icon: 'success'
        });
      } else {
        uni.showToast({
          title: `已关闭${type}权限`,
          icon: 'none'
        });
      }
    },
    toggleNotification(type) {
      this.notifications[type] = !this.notifications[type];
      console.log(`${type}通知状态:`, this.notifications[type]);
      
      // 提示用户
      if (this.notifications[type]) {
        uni.showToast({
          title: `已开启${type}通知`,
          icon: 'success'
        });
      } else {
        uni.showToast({
          title: `已关闭${type}通知`,
          icon: 'none'
        });
      }
    },
    toggleOtherSetting(type) {
      this.otherSettings[type] = !this.otherSettings[type];
      console.log(`${type}设置状态:`, this.otherSettings[type]);
      
      // 提示用户
      if (type === 'darkMode') {
        if (this.otherSettings[type]) {
          uni.showToast({
            title: '已切换到深色模式',
            icon: 'none'
          });
        } else {
          uni.showToast({
            title: '已切换到浅色模式',
            icon: 'none'
          });
        }
      }
    },
    goToPrivacyPolicy() {
      uni.showToast({
        title: '隐私政策功能开发中',
        icon: 'none'
      });
    },
    goToTerms() {
      uni.showToast({
        title: '用户协议功能开发中',
        icon: 'none'
      });
    },
    clearCache() {
      uni.showModal({
        title: '提示',
        content: '确定要清除缓存吗？',
        success: (res) => {
          if (res.confirm) {
            // 这里可以添加清除缓存的逻辑
            uni.showToast({
              title: '缓存已清除',
              icon: 'success'
            });
          }
        }
      });
    },
    checkUpdate() {
      uni.showLoading({
        title: '检查更新中...'
      });
      
      // 模拟检查更新
      setTimeout(() => {
        uni.hideLoading();
        uni.showToast({
          title: '当前已是最新版本',
          icon: 'none'
        });
      }, 1500);
    },
    goToFeedback() {
      uni.showToast({
        title: '意见反馈功能开发中',
        icon: 'none'
      });
    }
  }
}
</script>

<style>
.settings-container {
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

/* 设置列表样式 */
.section-title {
  font-size: 32rpx;
  font-weight: bold;
  color: #ffffff;
  margin: 30rpx 30rpx 20rpx;
}

.settings-list {
  margin: 0 30rpx 30rpx;
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: 20rpx;
  overflow: hidden;
}

.settings-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 30rpx;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.settings-item:last-child {
  border-bottom: none;
}

.settings-left {
  display: flex;
  align-items: center;
}

.settings-icon {
  font-size: 36rpx;
  margin-right: 20rpx;
}

.settings-text {
  font-size: 30rpx;
  color: #ffffff;
}

.settings-arrow {
  font-size: 30rpx;
  color: rgba(255, 255, 255, 0.6);
}
</style>