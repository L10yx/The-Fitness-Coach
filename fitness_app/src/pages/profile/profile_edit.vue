<template>
  <view class="edit-profile-container">
    <!-- 顶部导航栏 -->
    <view class="nav-bar">
      <view class="back-button" @click="goBack">
        <text class="back-icon">←</text>
      </view>
      <text class="page-title">编辑个人资料</text>
      <view class="save-button" @click="saveProfile">
        <text class="save-text">保存</text>
      </view>
    </view>
    
    <!-- 头像编辑区域 -->
    <view class="avatar-edit-section">
      <view class="avatar-large">
        <text class="avatar-text">{{ getAvatarText() }}</text>
      </view>
      <text class="avatar-hint">点击更换头像（开发中）</text>
    </view>
    
    <!-- 表单区域 -->
    <view class="form-container">
      <view class="form-group">
        <view class="form-item">
          <text class="form-label">昵称</text>
          <input class="form-input" type="text" v-model="userProfile.nickname" placeholder="请输入昵称" />
        </view>
        
        <view class="form-item">
          <text class="form-label">性别</text>
          <view class="gender-selector">
            <view 
              class="gender-option" 
              :class="{ 'gender-selected': userProfile.gender === '男' }"
              @click="userProfile.gender = '男'"
            >
              <text>男</text>
            </view>
            <view 
              class="gender-option" 
              :class="{ 'gender-selected': userProfile.gender === '女' }"
              @click="userProfile.gender = '女'"
            >
              <text>女</text>
            </view>
          </view>
        </view>
        
        <view class="form-item">
          <text class="form-label">年龄</text>
          <input class="form-input" type="number" v-model="userProfile.age" placeholder="请输入年龄" />
        </view>
      </view>
      
      <view class="section-title">健身数据</view>
      <view class="form-group">
        <view class="form-item">
          <text class="form-label">身高 (cm)</text>
          <input class="form-input" type="digit" v-model="userProfile.height" placeholder="请输入身高" />
        </view>
        
        <view class="form-item">
          <text class="form-label">体重 (kg)</text>
          <input class="form-input" type="digit" v-model="userProfile.weight" placeholder="请输入体重" />
        </view>
        
        <view class="form-item">
          <text class="form-label">健身目标</text>
          <picker 
            class="form-picker" 
            mode="selector" 
            :range="fitnessGoals" 
            @change="onGoalChange"
            :value="goalIndex"
          >
            <view class="picker-value">{{ userProfile.fitnessGoal || '请选择健身目标' }}</view>
          </picker>
        </view>
        
        <view class="form-item">
          <text class="form-label">健身经验</text>
          <picker 
            class="form-picker" 
            mode="selector" 
            :range="experienceLevels" 
            @change="onExperienceChange"
            :value="experienceIndex"
          >
            <view class="picker-value">{{ userProfile.experienceLevel || '请选择健身经验' }}</view>
          </picker>
        </view>
      </view>
      
      <view class="section-title">个人简介</view>
      <view class="form-group">
        <view class="form-item">
          <textarea 
            class="form-textarea" 
            v-model="userProfile.bio" 
            placeholder="介绍一下自己吧..."
            maxlength="200"
          />
          <text class="textarea-counter">{{ userProfile.bio.length }}/200</text>
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
      // 用户资料数据
      userProfile: {
        nickname: '',
        gender: '',
        age: '',
        height: '',
        weight: '',
        fitnessGoal: '',
        experienceLevel: '',
        bio: ''
      },
      // 健身目标选项
      fitnessGoals: ['增肌', '减脂', '塑形', '增强体能', '保持健康'],
      // 健身经验选项
      experienceLevels: ['初学者', '有一定基础', '进阶水平', '专业水平']
    }
  },
  computed: {
    // 计算健身目标在数组中的索引
    goalIndex() {
      return this.fitnessGoals.findIndex(goal => goal === this.userProfile.fitnessGoal);
    },
    // 计算健身经验在数组中的索引
    experienceIndex() {
      return this.experienceLevels.findIndex(level => level === this.userProfile.experienceLevel);
    }
  },
  onLoad() {
    // 页面加载时获取用户资料
    this.loadUserProfile();
  },
  methods: {
    // 返回上一页
    goBack() {
      uni.navigateBack();
    },
    
    // 加载用户资料
    loadUserProfile() {
      const userInfo = userManager.getCurrentUser();
      if (!userInfo) {
        uni.showToast({
          title: '请先登录',
          icon: 'none'
        });
        return;
      }
      
      // 获取用户详细资料
      const profile = userManager.getUserProfile();
      if (profile) {
        this.userProfile = { ...profile };
      } else {
        // 如果没有详细资料，使用基本用户信息初始化
        this.userProfile.nickname = userInfo.nickname || userInfo.username;
      }
    },
    
    // 保存用户资料
    saveProfile() {
      // 表单验证
      if (!this.userProfile.nickname) {
        uni.showToast({
          title: '请输入昵称',
          icon: 'none'
        });
        return;
      }
      
      // 保存用户资料
      const success = userManager.saveUserProfile(this.userProfile);
      
      if (success) {
        uni.showToast({
          title: '保存成功',
          icon: 'success'
        });
        
        // 更新当前用户的昵称
        const currentUser = userManager.getCurrentUser();
        if (currentUser) {
          currentUser.nickname = this.userProfile.nickname;
          userManager.saveUserInfo(currentUser);
        }
        
        // 返回上一页
        setTimeout(() => {
          uni.navigateBack();
        }, 1500);
      } else {
        uni.showToast({
          title: '保存失败',
          icon: 'none'
        });
      }
    },
    
    // 健身目标选择器变化事件
    onGoalChange(e) {
      const index = e.detail.value;
      this.userProfile.fitnessGoal = this.fitnessGoals[index];
    },
    
    // 健身经验选择器变化事件
    onExperienceChange(e) {
      const index = e.detail.value;
      this.userProfile.experienceLevel = this.experienceLevels[index];
    },
    
    // 获取头像文本
    getAvatarText() {
      if (this.userProfile.nickname) {
        return this.userProfile.nickname.charAt(0);
      }
      return '用户';
    }
  }
}
</script>

<style>
.edit-profile-container {
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

.save-button {
  width: 100rpx;
  height: 60rpx;
  background-color: #FF4785;
  border-radius: 30rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}

.save-text {
  font-size: 28rpx;
  color: #ffffff;
}

/* 头像编辑区域 */
.avatar-edit-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 40rpx 0;
}

.avatar-large {
  width: 160rpx;
  height: 160rpx;
  border-radius: 80rpx;
  background: linear-gradient(135deg, #FF4785 0%, #FF8D4E 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 20rpx;
}

.avatar-text {
  font-size: 60rpx;
  color: #ffffff;
  font-weight: bold;
}

.avatar-hint {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
}

/* 表单区域 */
.form-container {
  padding: 0 30rpx;
}

.section-title {
  font-size: 32rpx;
  font-weight: bold;
  color: #ffffff;
  margin: 30rpx 0 20rpx;
}

.form-group {
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: 20rpx;
  overflow: hidden;
  margin-bottom: 30rpx;
}

.form-item {
  padding: 30rpx;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  display: flex;
  align-items: center;
}

.form-item:last-child {
  border-bottom: none;
}

.form-label {
  width: 180rpx;
  font-size: 28rpx;
  color: rgba(255, 255, 255, 0.8);
}

.form-input {
  flex: 1;
  height: 60rpx;
  font-size: 28rpx;
  color: #ffffff;
  background-color: transparent;
}

.form-textarea {
  width: 100%;
  height: 200rpx;
  font-size: 28rpx;
  color: #ffffff;
  background-color: transparent;
  padding: 20rpx 0;
  line-height: 1.5;
}

.textarea-counter {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.4);
  text-align: right;
  margin-top: 10rpx;
}

/* 性别选择器 */
.gender-selector {
  display: flex;
  flex: 1;
}

.gender-option {
  flex: 1;
  height: 60rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 10rpx;
  margin-right: 20rpx;
}

.gender-option:last-child {
  margin-right: 0;
}

.gender-selected {
  background-color: rgba(255, 71, 133, 0.3);
  border: 1px solid #FF4785;
}

/* 选择器样式 */
.form-picker {
  flex: 1;
  height: 60rpx;
}

.picker-value {
  height: 60rpx;
  line-height: 60rpx;
  font-size: 28rpx;
  color: #ffffff;
}
</style>