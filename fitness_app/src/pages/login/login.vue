<template>
  <view class="login-container">
    <view class="logo-container">
      <image class="logo" src="/static/fitness_icon.png"></image>
      <text class="app-name">FitLife</text>
    </view>
    
    <view class="form-container">
      <view class="input-group">
        <text class="input-label">用户名</text>
        <input 
          class="input-field" 
          type="text" 
          v-model="username" 
          placeholder="请输入用户名" 
          placeholder-style="color: rgba(255, 255, 255, 0.4);"
        />
      </view>
      
      <view class="input-group">
        <text class="input-label">密码</text>
        <input 
          class="input-field" 
          type="password" 
          v-model="password" 
          placeholder="请输入密码" 
          placeholder-style="color: rgba(255, 255, 255, 0.4);"
        />
      </view>
      
      <button class="login-btn" @click="handleLogin">登录</button>
      
      <view class="options-container">
        <text class="option-text" @click="forgotPassword">忘记密码?</text>
        <text class="option-text" @click="goToRegister">注册账号</text>
      </view>
    </view>
    
    <view class="social-login">
      <text class="divider-text">或使用以下方式登录</text>
      <view class="social-icons">
        <view class="social-icon" @click="socialLogin('wechat')">
          <text class="icon-text">微信</text>
        </view>
        <view class="social-icon" @click="socialLogin('apple')">
          <text class="icon-text">Apple</text>
        </view>
      </view>
    </view>
  </view>
</template>

<script>
import userManager from '@/utils/userManager.js';

export default {
  data() {
    return {
      username: '',
      password: ''
    }
  },
  methods: {
    handleLogin() {
      // 登录逻辑
      if (!this.username || !this.password) {
        uni.showToast({
          title: '请输入用户名和密码',
          icon: 'none'
        });
        return;
      }
      
      // 显示加载中
      uni.showLoading({
        title: '登录中...'
      });
      
      // 使用userManager进行登录
      userManager.login(this.username, this.password)
        .then(result => {
          uni.hideLoading();
          
          if (result.success) {
            uni.showToast({
              title: '登录成功',
              icon: 'success'
            });
            
            // 登录成功后跳转到主界面
            setTimeout(() => {
              uni.reLaunch({
                url: '/pages/main/main'
              });
            }, 1500);
          } else {
            uni.showToast({
              title: result.message || '登录失败',
              icon: 'none'
            });
          }
        })
        .catch(error => {
          uni.hideLoading();
          console.error('登录出错:', error);
          
          uni.showToast({
            title: '登录失败，请重试',
            icon: 'none'
          });
        });
    },
    forgotPassword() {
      uni.showToast({
        title: '忘记密码功能开发中',
        icon: 'none'
      });
    },
    goToRegister() {
      uni.showToast({
        title: '注册功能开发中',
        icon: 'none'
      });
    },
    socialLogin(type) {
      uni.showToast({
        title: `${type}登录功能开发中`,
        icon: 'none'
      });
    }
  }
}
</script>

<style>
.login-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  padding: 40rpx;
  background-color: #121212;
  color: #ffffff;
}

.logo-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 80rpx;
  margin-bottom: 60rpx;
}

.logo {
  width: 160rpx;
  height: 160rpx;
  border-radius: 40rpx;
}

.app-name {
  font-size: 48rpx;
  font-weight: bold;
  margin-top: 20rpx;
  background: linear-gradient(90deg, #FF4785 0%, #FF8D4E 100%);
  -webkit-background-clip: text;
  color: transparent;
}

.form-container {
  width: 100%;
  margin-bottom: 60rpx;
}

.input-group {
  margin-bottom: 30rpx;
}

.input-label {
  font-size: 28rpx;
  color: rgba(255, 255, 255, 0.8);
  margin-bottom: 10rpx;
  display: block;
}

.input-field {
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 16rpx;
  height: 90rpx;
  padding: 0 30rpx;
  color: #ffffff;
  font-size: 30rpx;
  width: 100%;
  box-sizing: border-box;
}

.login-btn {
  background: linear-gradient(90deg, #FF4785 0%, #FF8D4E 100%);
  border-radius: 16rpx;
  height: 90rpx;
  line-height: 90rpx;
  color: #ffffff;
  font-size: 32rpx;
  font-weight: bold;
  margin-top: 50rpx;
  border: none;
}

.options-container {
  display: flex;
  justify-content: space-between;
  margin-top: 30rpx;
}

.option-text {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.6);
}

.social-login {
  margin-top: auto;
  margin-bottom: 40rpx;
}

.divider-text {
  text-align: center;
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.4);
  position: relative;
  margin: 40rpx 0;
}

.divider-text::before,
.divider-text::after {
  content: '';
  position: absolute;
  top: 50%;
  width: 25%;
  height: 1px;
  background-color: rgba(255, 255, 255, 0.2);
}

.divider-text::before {
  left: 0;
}

.divider-text::after {
  right: 0;
}

.social-icons {
  display: flex;
  justify-content: center;
  gap: 60rpx;
  margin-top: 30rpx;
}

.social-icon {
  width: 120rpx;
  height: 120rpx;
  border-radius: 60rpx;
  background-color: rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
}

.icon-text {
  font-size: 26rpx;
  color: #ffffff;
}
</style>