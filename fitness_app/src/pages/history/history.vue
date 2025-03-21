<template>
  <view class="history-container">
    <!-- 顶部导航栏 -->
    <view class="nav-bar">
      <view class="back-button" @click="goBack">
        <text class="back-icon">←</text>
      </view>
      <text class="page-title">历史记录</text>
      <view class="add-button" @click="showAddModal">
        <text class="add-icon">+</text>
      </view>
    </view>
    
    <!-- 记录列表 -->
    <view class="records-list" v-if="records.length > 0">
      <view class="record-item" v-for="(record, index) in records" :key="index">
        <view class="record-info">
          <text class="record-type">{{ record.type }}</text>
          <text class="record-date">{{ record.date }}</text>
        </view>
        <view class="record-details">
          <text class="record-count">{{ record.count }} 次</text>
          <text class="record-duration" v-if="record.duration">{{ record.duration }} 分钟</text>
        </view>
        <view class="delete-button" @click="deleteRecord(index)">
          <text class="delete-icon">×</text>
        </view>
      </view>
    </view>
    
    <!-- 空状态 -->
    <view class="empty-state" v-else>
      <text class="empty-icon">📊</text>
      <text class="empty-text">暂无历史记录</text>
      <text class="empty-subtext">点击右上角添加按钮开始记录</text>
    </view>
    
    <!-- 添加记录弹窗 -->
    <view class="modal-overlay" v-if="showModal" @click="hideModal"></view>
    <view class="add-modal" v-if="showModal">
      <text class="modal-title">添加记录</text>
      
      <view class="form-item">
        <text class="form-label">运动类型</text>
        <view class="type-selector">
          <view 
            v-for="type in exerciseTypes" 
            :key="type.value"
            :class="['type-option', newRecord.type === type.value ? 'selected' : '']"
            @click="selectType(type.value)"
          >
            <text class="type-text">{{ type.label }}</text>
          </view>
        </view>
      </view>
      
      <view class="form-item">
        <text class="form-label">日期</text>
        <input class="form-input" type="date" v-model="newRecord.date" />
      </view>
      
      <view class="form-item">
        <text class="form-label">次数</text>
        <input class="form-input" type="number" v-model="newRecord.count" placeholder="请输入次数" />
      </view>
      
      <view class="form-item">
        <text class="form-label">时长 (分钟)</text>
        <input class="form-input" type="number" v-model="newRecord.duration" placeholder="可选" />
      </view>
      
      <view class="modal-buttons">
        <view class="cancel-button" @click="hideModal">
          <text class="button-text">取消</text>
        </view>
        <view class="confirm-button" @click="addRecord">
          <text class="button-text">确认</text>
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
      records: [],
      showModal: false,
      newRecord: {
        type: 'pullup',
        date: this.formatDate(new Date()),
        count: '',
        duration: ''
      },
      exerciseTypes: [
        { label: '引体向上', value: 'pullup' },
        { label: '俯卧撑', value: 'pushup' },
        { label: '卷腹', value: 'crunch' }
      ]
    }
  },
  onShow() {
    // 每次页面显示时加载最新记录
    this.loadRecords();
  },
  
  methods: {
    loadRecords() {
      // 从用户管理器获取记录
      this.records = userManager.getUserExerciseHistory();
    },
    goBack() {
      uni.navigateBack();
    },
    showAddModal() {
      // 重置表单
      this.newRecord = {
        type: 'pullup',
        date: this.formatDate(new Date()),
        count: '',
        duration: ''
      };
      this.showModal = true;
    },
    hideModal() {
      this.showModal = false;
    },
    selectType(type) {
      this.newRecord.type = type;
    },
    formatDate(date) {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      return `${year}-${month}-${day}`;
    },
    addRecord() {
      // 验证输入
      if (!this.newRecord.count) {
        uni.showToast({
          title: '请输入次数',
          icon: 'none'
        });
        return;
      }
      
      // 创建记录对象
      const record = {
        type: this.getExerciseName(this.newRecord.type),
        date: this.newRecord.date,
        count: parseInt(this.newRecord.count),
        duration: this.newRecord.duration ? parseInt(this.newRecord.duration) : null
      };
      
      // 使用userManager添加记录
      const success = userManager.addExerciseRecord(record);
      
      if (success) {
        // 重新加载记录
        this.loadRecords();
        
        // 关闭弹窗
        this.hideModal();
        
        uni.showToast({
          title: '记录已添加',
          icon: 'success'
        });
      } else {
        uni.showToast({
          title: '添加失败，请先登录',
          icon: 'none'
        });
      }
    },
    deleteRecord(index) {
      uni.showModal({
        title: '确认删除',
        content: '确定要删除这条记录吗？',
        success: (res) => {
          if (res.confirm) {
            const recordId = this.records[index].id;
            
            // 使用userManager删除记录
            const success = userManager.deleteExerciseRecord(recordId);
            
            if (success) {
              // 重新加载记录
              this.loadRecords();
              
              uni.showToast({
                title: '记录已删除',
                icon: 'success'
              });
            } else {
              uni.showToast({
                title: '删除失败，请先登录',
                icon: 'none'
              });
            }
          }
        }
      });
    },
    getExerciseName(type) {
      const exercise = this.exerciseTypes.find(item => item.value === type);
      return exercise ? exercise.label : type;
    }
  }
}
</script>

<style>
.history-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: #121212;
  color: #ffffff;
  padding: 40rpx;
  box-sizing: border-box;
}

/* 导航栏样式 */
.nav-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 40rpx;
}

.back-button, .add-button {
  width: 70rpx;
  height: 70rpx;
  border-radius: 35rpx;
  background-color: rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
}

.back-icon {
  font-size: 36rpx;
  color: #ffffff;
}

.add-icon {
  font-size: 36rpx;
  color: #ffffff;
}

.page-title {
  font-size: 36rpx;
  font-weight: bold;
  color: #ffffff;
}

/* 记录列表样式 */
.records-list {
  margin-bottom: 40rpx;
}

.record-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: rgba(255, 255, 255, 0.08);
  border-radius: 20rpx;
  padding: 30rpx;
  margin-bottom: 20rpx;
  position: relative;
  overflow: hidden;
}

.record-item::after {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  width: 6rpx;
  height: 100%;
  background: linear-gradient(90deg, #FF4785 0%, #FF8D4E 100%);
}

.record-info {
  display: flex;
  flex-direction: column;
}

.record-type {
  font-size: 32rpx;
  font-weight: bold;
  color: #ffffff;
  margin-bottom: 10rpx;
}

.record-date {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
}

.record-details {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}

.record-count {
  font-size: 32rpx;
  font-weight: bold;
  color: #FF4785;
  margin-bottom: 10rpx;
}

.record-duration {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
}

.delete-button {
  width: 60rpx;
  height: 60rpx;
  border-radius: 30rpx;
  background-color: rgba(255, 71, 133, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 20rpx;
}

.delete-icon {
  font-size: 32rpx;
  color: #FF4785;
}

/* 空状态样式 */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin-top: 100rpx;
}

.empty-icon {
  font-size: 80rpx;
  margin-bottom: 30rpx;
}

.empty-text {
  font-size: 36rpx;
  font-weight: bold;
  color: #ffffff;
  margin-bottom: 10rpx;
}

.empty-subtext {
  font-size: 28rpx;
  color: rgba(255, 255, 255, 0.6);
}

/* 弹窗样式 */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.7);
  z-index: 999;
}

.add-modal {
  position: fixed;
  left: 50rpx;
  right: 50rpx;
  bottom: 50rpx;
  background-color: #1E1E1E;
  border-radius: 20rpx;
  padding: 40rpx;
  z-index: 1000;
}

.modal-title {
  font-size: 36rpx;
  font-weight: bold;
  color: #ffffff;
  margin-bottom: 30rpx;
  text-align: center;
}

.form-item {
  margin-bottom: 30rpx;
}

.form-label {
  font-size: 28rpx;
  color: rgba(255, 255, 255, 0.8);
  margin-bottom: 10rpx;
  display: block;
}

.form-input {
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 10rpx;
  padding: 20rpx;
  color: #ffffff;
  font-size: 28rpx;
}

.type-selector {
  display: flex;
  justify-content: space-between;
  margin-top: 10rpx;
}

.type-option {
  flex: 1;
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 10rpx;
  padding: 20rpx 0;
  margin: 0 10rpx;
  text-align: center;
}

.type-option:first-child {
  margin-left: 0;
}

.type-option:last-child {
  margin-right: 0;
}

.type-option.selected {
  background-color: rgba(255, 71, 133, 0.3);
  border: 1px solid #FF4785;
}

.type-text {
  font-size: 28rpx;
  color: #ffffff;
}

.modal-buttons {
  display: flex;
  justify-content: space-between;
  margin-top: 40rpx;
}

.cancel-button, .confirm-button {
  flex: 1;
  padding: 20rpx 0;
  border-radius: 10rpx;
  text-align: center;
}

.cancel-button {
  background-color: rgba(255, 255, 255, 0.1);
  margin-right: 20rpx;
}

.confirm-button {
  background: linear-gradient(90deg, #FF4785 0%, #FF8D4E 100%);
}

.button-text {
  font-size: 30rpx;
  color: #ffffff;
  font-weight: bold;
}
</style>