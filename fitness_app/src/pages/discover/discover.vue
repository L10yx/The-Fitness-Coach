<template>
  <view class="discover-container">
    <!-- 顶部导航栏 -->
    <view class="nav-bar">
      <text class="page-title">发现</text>
      <view class="post-btn" @click="showPostModal">
        <text class="post-icon">+</text>
      </view>
    </view>
    
    <!-- 内容区域 -->
    <scroll-view scroll-y class="post-list">
      <!-- 发帖输入框 -->
      <view v-if="showingPostInput" class="post-input-container">
        <view class="post-input-header">
          <text class="post-input-title">发布新动态</text>
          <view class="close-btn" @click="hidePostModal">
            <text class="close-icon">×</text>
          </view>
        </view>
        <textarea 
          class="post-textarea" 
          placeholder="分享你的健身心得..." 
          v-model="newPostContent"
          maxlength="280"
        />
        <view class="post-input-footer">
          <text class="char-count">{{ newPostContent.length }}/280</text>
          <view class="submit-btn" @click="submitPost">
            <text class="submit-text">发布</text>
          </view>
        </view>
      </view>
      
      <!-- 帖子列表 -->
      <post-card 
        v-for="(post, index) in posts" 
        :key="index"
        :post="post"
        :post-index="index"
        @update-post="updatePost"
      />
    </scroll-view>
    
    <!-- 底部导航栏 -->
    <view class="tab-bar">
      <view class="tab-item active">
        <text class="tab-icon">⚪</text>
        <text class="tab-text">发现</text>
      </view>
      <view class="tab-item" @click="goToMain">
        <text class="tab-icon">⚪</text>
        <text class="tab-text">开始</text>
      </view>
      <view class="tab-item" @click="goToStats">
        <text class="tab-icon">⚪</text>
        <text class="tab-text">统计</text>
      </view>
    </view>
  </view>
</template>

<script>
import PostCard from '@/components/PostCard.vue';

export default {
  components: {
    PostCard
  },
  data() {
    return {
      showingPostInput: false,
      newPostContent: '',
      newCommentText: '',
      posts: [
        {
          author: '健身达人',
          time: '10分钟前',
          content: '今天完成了30个引体向上，感觉状态越来越好了！💪 #健身打卡',
          likes: 24,
          liked: false,
          bookmarked: false,
          showingComments: false,
          comments: [
            {
              author: '运动爱好者',
              time: '8分钟前',
              content: '太厉害了！请问有什么训练技巧吗？'
            },
            {
              author: '新手小白',
              time: '5分钟前',
              content: '我才能做5个，继续努力！'
            }
          ]
        },
        {
          author: '营养师Mike',
          time: '1小时前',
          content: '健身饮食小贴士：训练后30分钟内补充蛋白质和碳水化合物，有助于肌肉恢复和生长。今天分享我的训练后shake配方：香蕉+蛋白粉+牛奶+花生酱，简单又美味！',
          likes: 56,
          liked: true,
          bookmarked: true,
          showingComments: false,
          comments: [
            {
              author: '健身小白',
              time: '50分钟前',
              content: '感谢分享！请问蛋白粉推荐哪个牌子？'
            },
            {
              author: '饮食达人',
              time: '30分钟前',
              content: '我也喜欢这个配方，有时候加点燕麦更赞！'
            },
            {
              author: '减脂君',
              time: '10分钟前',
              content: '请问热量大概多少？'
            }
          ]
        },
        {
          author: '瑜伽教练Linda',
          time: '3小时前',
          content: '坚持晨练一个月的变化：精力更充沛，睡眠质量提高，心情也变好了。晨间锻炼真的能改变一天的状态！大家都是什么时候锻炼的呢？',
          likes: 42,
          liked: false,
          bookmarked: false,
          showingComments: false,
          comments: [
            {
              author: '夜猫子',
              time: '2小时前',
              content: '我喜欢晚上锻炼，白天没时间啊'
            },
            {
              author: '早起鸟',
              time: '1小时前',
              content: '早晨锻炼+1，感觉一整天都很有活力'
            }
          ]
        }
      ]
    }
  },
  methods: {
    showPostModal() {
      this.showingPostInput = true;
      this.newPostContent = '';
    },
    hidePostModal() {
      this.showingPostInput = false;
    },
    submitPost() {
      if (this.newPostContent.trim() === '') {
        uni.showToast({
          title: '内容不能为空',
          icon: 'none'
        });
        return;
      }
      
      // 添加新帖子
      this.posts.unshift({
        author: '健身哥', // 当前用户名
        time: '刚刚',
        content: this.newPostContent,
        likes: 0,
        liked: false,
        bookmarked: false,
        showingComments: false,
        comments: []
      });
      
      // 清空输入框并隐藏
      this.newPostContent = '';
      this.hidePostModal();
      
      uni.showToast({
        title: '发布成功',
        icon: 'success'
      });
    },
    updatePost(index, updatedPost) {
      // 更新帖子数据
      this.posts[index] = updatedPost;
    },
    goToMain() {
      uni.redirectTo({
        url: '/pages/main/main'
      });
    },
    goToStats() {
      uni.redirectTo({
        url: '/pages/stats/stats'
      });
    }
  }
}
</script>

<style>
.discover-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: #121212;
  color: #ffffff;
  position: relative;
  padding-bottom: 140rpx; /* 为底部导航栏留出空间 */
}

/* 导航栏样式 */
.nav-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 90rpx;
  padding: 40rpx 30rpx 20rpx;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.page-title {
  font-size: 36rpx;
  font-weight: bold;
  color: #ffffff;
}

.post-btn {
  width: 60rpx;
  height: 60rpx;
  border-radius: 30rpx;
  background: linear-gradient(135deg, #FF4785 0%, #FF8D4E 100%);
  display: flex;
  align-items: center;
  justify-content: center;
}

.post-icon {
  font-size: 40rpx;
  color: #ffffff;
  font-weight: bold;
}

/* 帖子列表样式 */
.post-list {
  flex: 1;
  padding: 0 30rpx;
}

/* 发帖输入框样式 */
.post-input-container {
  background-color: rgba(255, 255, 255, 0.08);
  border-radius: 20rpx;
  padding: 30rpx;
  margin: 30rpx 0;
}

.post-input-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20rpx;
}

.post-input-title {
  font-size: 32rpx;
  font-weight: bold;
  color: #ffffff;
}

.close-btn {
  width: 50rpx;
  height: 50rpx;
  border-radius: 25rpx;
  background-color: rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-icon {
  font-size: 36rpx;
  color: rgba(255, 255, 255, 0.8);
}

.post-textarea {
  width: 100%;
  height: 200rpx;
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: 16rpx;
  padding: 20rpx;
  color: #ffffff;
  font-size: 28rpx;
  margin-bottom: 20rpx;
}

.post-input-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.char-count {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
}

.submit-btn {
  background: linear-gradient(135deg, #FF4785 0%, #FF8D4E 100%);
  border-radius: 30rpx;
  padding: 10rpx 30rpx;
}

.submit-text {
  font-size: 28rpx;
  color: #ffffff;
  font-weight: bold;
}

/* 底部导航栏样式 */
.tab-bar {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  height: 120rpx;
  background-color: rgba(18, 18, 18, 0.95);
  display: flex;
  justify-content: space-around;
  align-items: center;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  padding-bottom: env(safe-area-inset-bottom);
}

.tab-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 33.33%;
  height: 100%;
}

.tab-icon {
  font-size: 44rpx;
  color: rgba(255, 255, 255, 0.6);
  margin-bottom: 8rpx;
  transition: color 0.3s ease;
}

.tab-text {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
}

.tab-item.active .tab-icon,
.tab-item.active .tab-text {
  color: #FF4785;
}
</style>