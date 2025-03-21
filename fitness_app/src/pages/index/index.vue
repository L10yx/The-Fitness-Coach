<template>
  <view class="content">
    <view class="header">
      <image class="logo" src="/static/fitness_icon.png"></image>
      <text class="app-name">FitLife</text>
    </view>
    <view class="text-area">
      <text class="title">{{ title }}</text>
      <text class="subtitle">开启您的健身之旅</text>
    </view>
    <button class="login-btn" hover-class="btn-hover" @click="goToLogin">开始体验</button>
  </view>
</template>

<script>
export default {
  data() {
    return {
      title: '智能健身助手',
    }
  },
  onLoad() {},
  methods: {
    goToLogin() {
      uni.navigateTo({
        url: '/pages/login/login'
      });
    }
  },
}
</script>

<style>
.content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  min-height: 100vh;
  background-color: #121212;
  color: #ffffff;
  padding: 60rpx 40rpx;
  box-sizing: border-box;
}

.header {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 80rpx;
}

.logo {
  height: 180rpx;
  width: 180rpx;
  border-radius: 40rpx;
  box-shadow: 0 10rpx 30rpx rgba(255, 71, 133, 0.2);
}

.app-name {
  font-size: 48rpx;
  font-weight: bold;
  margin-top: 20rpx;
  background: linear-gradient(90deg, #FF4785 0%, #FF8D4E 100%);
  -webkit-background-clip: text;
  color: transparent;
}

.text-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 60rpx 0;
}

.title {
  font-size: 40rpx;
  font-weight: bold;
  color: #ffffff;
  margin-bottom: 20rpx;
}

.subtitle {
  font-size: 30rpx;
  color: rgba(255, 255, 255, 0.7);
}

.login-btn {
  width: 80%;
  background: linear-gradient(90deg, #FF4785 0%, #FF8D4E 100%);
  color: #ffffff;
  border-radius: 16rpx;
  padding: 25rpx 0;
  font-size: 32rpx;
  font-weight: bold;
  margin-bottom: 80rpx;
  border: none;
  transition: all 0.3s ease;
}

.btn-hover {
  transform: scale(0.98);
  opacity: 0.9;
}
</style>
