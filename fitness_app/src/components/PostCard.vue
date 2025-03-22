<template>
  <view class="post-card">
    <view class="post-header">
      <view class="post-avatar">
        <text class="avatar-text">{{ post.author.charAt(0) }}</text>
      </view>
      <view class="post-author-info">
        <text class="post-author">{{ post.author }}</text>
        <text class="post-time">{{ post.time }}</text>
      </view>
    </view>
    <view class="post-content">
      <text class="post-text">{{ post.content }}</text>
    </view>
    <view class="post-actions">
      <view class="action-btn" @click="likePost">
        <text class="action-icon" :class="{ 'active': post.liked }">♥</text>
        <text class="action-count">{{ post.likes }}</text>
      </view>
      <view class="action-btn" @click="toggleComments">
        <text class="action-icon">💬</text>
        <text class="action-count">{{ post.comments.length }}</text>
      </view>
      <view class="action-btn" @click="bookmarkPost">
        <text class="action-icon" :class="{ 'active': post.bookmarked }">⭐</text>
      </view>
    </view>
    
    <!-- 评论区域 -->
    <view v-if="post.showingComments" class="comments-section">
      <view v-for="(comment, cIndex) in post.comments" :key="cIndex" class="comment-item">
        <view class="comment-avatar">
          <text class="avatar-text-small">{{ comment.author.charAt(0) }}</text>
        </view>
        <view class="comment-content">
          <view class="comment-header">
            <text class="comment-author">{{ comment.author }}</text>
            <text class="comment-time">{{ comment.time }}</text>
          </view>
          <text class="comment-text">{{ comment.content }}</text>
        </view>
      </view>
      
      <!-- 评论输入框 -->
      <view class="comment-input-container">
        <input 
          class="comment-input" 
          placeholder="添加评论..." 
          v-model="newCommentText"
          @confirm="addComment"
        />
        <view class="send-btn" @click="addComment">
          <text class="send-icon">↑</text>
        </view>
      </view>
    </view>
  </view>
</template>

<script>
export default {
  props: {
    post: {
      type: Object,
      required: true
    },
    postIndex: {
      type: Number,
      required: true
    }
  },
  data() {
    return {
      newCommentText: ''
    }
  },
  methods: {
    likePost() {
      // 创建一个副本以避免直接修改props
      const updatedPost = JSON.parse(JSON.stringify(this.post));
      updatedPost.liked = !updatedPost.liked;
      updatedPost.likes += updatedPost.liked ? 1 : -1;
      this.$emit('update-post', this.postIndex, updatedPost);
    },
    bookmarkPost() {
      const updatedPost = JSON.parse(JSON.stringify(this.post));
      updatedPost.bookmarked = !updatedPost.bookmarked;
      this.$emit('update-post', this.postIndex, updatedPost);
      
      if (updatedPost.bookmarked) {
        uni.showToast({
          title: '已加入收藏',
          icon: 'success'
        });
      } else {
        uni.showToast({
          title: '已取消收藏',
          icon: 'none'
        });
      }
    },
    toggleComments() {
      // 切换评论区显示状态
      const updatedPost = JSON.parse(JSON.stringify(this.post));
      updatedPost.showingComments = !updatedPost.showingComments;
      this.$emit('update-post', this.postIndex, updatedPost);
      this.newCommentText = '';
    },
    addComment() {
      if (this.newCommentText.trim() === '') {
        uni.showToast({
          title: '评论不能为空',
          icon: 'none'
        });
        return;
      }
      
      // 添加新评论
      const updatedPost = JSON.parse(JSON.stringify(this.post));
      updatedPost.comments.push({
        author: '健身哥', // 当前用户名
        time: '刚刚',
        content: this.newCommentText
      });
      
      // 更新帖子并清空输入框
      this.$emit('update-post', this.postIndex, updatedPost);
      this.newCommentText = '';
    }
  }
}
</script>

<style>
.post-card {
  background-color: rgba(255, 255, 255, 0.08);
  border-radius: 20rpx;
  padding: 30rpx;
  margin-bottom: 30rpx;
}

.post-header {
  display: flex;
  align-items: center;
  margin-bottom: 20rpx;
}

.post-avatar {
  width: 80rpx;
  height: 80rpx;
  border-radius: 40rpx;
  background: linear-gradient(135deg, #FF4785 0%, #FF8D4E 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 20rpx;
}

.avatar-text {
  font-size: 32rpx;
  color: #ffffff;
  font-weight: bold;
}

.post-author-info {
  flex: 1;
}

.post-author {
  font-size: 30rpx;
  font-weight: bold;
  color: #ffffff;
  margin-bottom: 6rpx;
  display: block;
}

.post-time {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
}

.post-content {
  margin-bottom: 20rpx;
}

.post-text {
  font-size: 28rpx;
  color: #ffffff;
  line-height: 1.5;
}

.post-actions {
  display: flex;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  padding-top: 20rpx;
}

.action-btn {
  display: flex;
  align-items: center;
  margin-right: 40rpx;
}

.action-icon {
  font-size: 36rpx;
  color: rgba(255, 255, 255, 0.6);
  margin-right: 10rpx;
}

.action-icon.active {
  color: #FF4785;
}

.action-count {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
}

/* 评论区样式 */
.comments-section {
  margin-top: 20rpx;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  padding-top: 20rpx;
}

.comment-item {
  display: flex;
  margin-bottom: 20rpx;
}

.comment-avatar {
  width: 60rpx;
  height: 60rpx;
  border-radius: 30rpx;
  background: rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 20rpx;
}

.avatar-text-small {
  font-size: 24rpx;
  color: #ffffff;
  font-weight: bold;
}

.comment-content {
  flex: 1;
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: 16rpx;
  padding: 16rpx;
}

.comment-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10rpx;
}

.comment-author {
  font-size: 26rpx;
  font-weight: bold;
  color: #ffffff;
}

.comment-time {
  font-size: 22rpx;
  color: rgba(255, 255, 255, 0.5);
}

.comment-text {
  font-size: 26rpx;
  color: #ffffff;
  line-height: 1.4;
}

.comment-input-container {
  display: flex;
  align-items: center;
  margin-top: 20rpx;
}

.comment-input {
  flex: 1;
  height: 70rpx;
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 35rpx;
  padding: 0 30rpx;
  color: #ffffff;
  font-size: 26rpx;
}

.send-btn {
  width: 70rpx;
  height: 70rpx;
  border-radius: 35rpx;
  background: linear-gradient(135deg, #FF4785 0%, #FF8D4E 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 20rpx;
}

.send-icon {
  font-size: 30rpx;
  color: #ffffff;
  font-weight: bold;
}
</style>