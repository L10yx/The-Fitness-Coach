<template>
  <view class="exercise-card" @click="startExercise">
    <view class="exercise-icon">
      <text class="icon-placeholder">{{ iconText }}</text>
    </view>
    <view class="exercise-info">
      <text class="exercise-name">{{ name }}</text>
      <text class="exercise-desc">{{ description }}</text>
    </view>
    <view class="start-btn">
      <text class="btn-text">开始</text>
    </view>
  </view>
</template>

<script>
export default {
  props: {
    name: {
      type: String,
      required: true
    },
    description: {
      type: String,
      default: '开始AI计数'
    },
    iconText: {
      type: String,
      default: '运动'
    },
    exerciseType: {
      type: String,
      required: true
    }
  },
  methods: {
    startExercise() {
      this.$emit('exercise-selected', this.exerciseType);
      // 后续可以添加导航到计数页面的逻辑
      uni.showToast({
        title: `开始${this.name}计数`,
        icon: 'none'
      });
    }
  }
}
</script>

<style>
.exercise-card {
  display: flex;
  align-items: center;
  background-color: rgba(255, 255, 255, 0.08);
  border-radius: 20rpx;
  padding: 30rpx;
  margin-bottom: 30rpx;
  position: relative;
  overflow: hidden;
}

.exercise-card::after {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  width: 6rpx;
  height: 100%;
  background: linear-gradient(90deg, #FF4785 0%, #FF8D4E 100%);
}

.exercise-icon {
  width: 100rpx;
  height: 100rpx;
  border-radius: 50rpx;
  background: rgba(255, 71, 133, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 30rpx;
}

.icon-placeholder {
  font-size: 32rpx;
  color: #FF4785;
  font-weight: bold;
}

.exercise-info {
  flex: 1;
}

.exercise-name {
  font-size: 34rpx;
  font-weight: bold;
  color: #ffffff;
  margin-bottom: 10rpx;
}

.exercise-desc {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.6);
}

.start-btn {
  background: linear-gradient(90deg, #FF4785 0%, #FF8D4E 100%);
  border-radius: 40rpx;
  padding: 15rpx 30rpx;
}

.btn-text {
  font-size: 28rpx;
  color: #ffffff;
  font-weight: bold;
}
</style>