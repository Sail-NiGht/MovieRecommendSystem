package com.zyf.offline

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix


//定义样例类

// 基于评分数据的LFM，只需要rating数据
//类名区别于MLlib定义的Rating样例类
case class MovieRating(uid: Int, mid: Int, score: Double, timestamp: Int )
//定义Mongo配置封装成的样例类
case class MongoConfig(uri:String, db:String)

// 定义一个基准推荐对象
case class Recommendation( mid: Int, score: Double )
// 定义基于预测评分的用户推荐列表
case class UserRecs( uid: Int, recs: Seq[Recommendation] )
// 定义基于LFM电影特征向量的电影相似度列表
case class MovieRecs( mid: Int, recs: Seq[Recommendation] )


object OfflineRecommmeder {

  // 定义数据集表名
  val MONGODB_RATING_COLLECTION = "Rating"
  // 定义Mongo内列表的表名
  val USER_RECS = "UserRecs"    //针对用户的推荐列表
  val MOVIE_RECS = "MovieRecs"
  // 定义常量：推荐数量
  val USER_MAX_RECOMMENDATION = 20

  //主函数
  def main(args: Array[String]): Unit = {
    //定义配置常量
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://localhost:27017/recommender",
      "mongo.db" -> "recommender"
    )
    //创建一个sparkConf对象
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("OfflineRecommender")
    // 创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._
    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    // 加载数据
    //从Mongo获取数据，并转化为RDD
    val ratingRDD = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]    //MovieRating样例类
      .rdd                //调用ALS算法参数要求RDD格式，转化成RDD
      .map( rating => ( rating.uid, rating.mid, rating.score ) )    // 去掉时间戳
      .cache()            //使数据持久化在内存
// 打印ratingRDD
//println("打印ratingRDD：")
//ratingRDD.collect()foreach(println)


    // 从 ratingRDD 数据中提取所有的uid和mid，并去重
    // 得到用户和电影的列表，用于构造空矩阵
    val userRDD = ratingRDD.map(_._1).distinct()  //得到所有用户uid
    val movieRDD = ratingRDD.map(_._2).distinct() //得到所有电影mid
    // 打印userRDD和movieRDD
//println("打印userRDD：")
//userRDD.collect()foreach(println)
//println("打印movieRDD：")
//movieRDD.collect()foreach(println)


    // 训练隐语义模型
    val trainData = ratingRDD.map( x => Rating(x._1, x._2, x._3) )  //转化为标准Rating类型
    //rank：隐特征向量维度K
    //iterations：迭代次数
    //lambda：正则化系数
    val (rank, iterations, lambda) = (50, 5, 0.1)  //声明常量
    val model = ALS.train(trainData, rank, iterations, lambda)
    //返回的model为MatrixFactorizationModel类型

    // 基于用户和电影的隐特征，计算预测评分，得到用户的推荐列表
    // 计算user和movie的笛卡尔积，得到一个空评分矩阵
    //  （即第一个RDD的每个项与第二个RDD的每个项连接）并将它们作为新的RDD返回
    val userMovies = userRDD.cartesian(movieRDD)
//打印空笛卡尔积：userMovies
//println("打印空笛卡尔积：userMovies")
//userMovies.collect()foreach(println)

    // 调用隐语义模型model的predict方法预测评分
    //返回结果为MLlib中的Rating类型的RDD：每个元素含有user、product、rating三个属性
    val preRatings = model.predict(userMovies)
//打印预测评分：preRatings
//println("打印预测评分：preRatings")
//preRatings.collect()foreach(println)


    //对预测评分排序，得到针对用户的推荐列表
    val userRecs = preRatings
      .filter(_.rating > 0)    // 过滤出评分大于0的项
      //将每一条数据的product和rating封装到一个元组
      .map(rating => ( rating.user, (rating.product, rating.rating) ) )
      .groupByKey() //将user值相同的数据聚类
      .map{
      //模式匹配：将每条数据转成定义好的样例类UserRecs( uid: Int, recs: Seq[Recommendation] )
      //每个uid截取recs中的前20个构造推荐列表Seq[Recommendation]（按第二个元素rating.rating降序排列）
      //uid: Int, recs: Seq[Recommendation]为Recommendation对象的列表
        case (uid, recs) => UserRecs( uid, recs.toList.sortWith(_._2>_._2).take(USER_MAX_RECOMMENDATION).map(x=>Recommendation(x._1, x._2)) )
      }
      .toDF()
//打印针对用户的推荐列表：userRecs
//println("打印针对用户的推荐列表：userRecs")
//userRecs.collect()foreach(println)

    //把数据存到MongoDB
    userRecs.write
      .option("uri", mongoConfig.uri)
      .option("collection", USER_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()




    // 基于电影隐特征，计算相似度矩阵，得到 “电影的相似度列表”

    //通过隐语义模型得到电影的 “特征矩阵”
    val movieFeatures = model.productFeatures.map{
      //每一条为mid+特征值数组:RDD[(Int, Array[Double])]
      //将特征值数组Array[Double]转化为DoubleMatrix类型，用于矩阵计算
      case (mid, features) => (mid, new DoubleMatrix(features))
    }
//打印电影的 “特征矩阵”：movieFeatures
//println("打印电影的 “特征矩阵”：movieFeatures")
//movieFeatures.collect()foreach(println)

    // 对所有电影两两计算它们的相似度，先做笛卡尔积
    val movieRecs = movieFeatures.cartesian(movieFeatures)
      .filter{
        // 把自己跟自己的配对过滤掉
        // a的mid 不能等于 b的mid
        case (a, b) => a._1 != b._1
              }
      .map{
        case (a, b) => {
          val simScore = this.consinSim(a._2, b._2) //计算a的特征值数组与b的特征值数组的余弦相似度
          //( a的mid, 元组( b的mid, a与b的特征余弦相似度 )
          ( a._1, ( b._1, simScore ) )
          }
      }
      //_._2._2对应所有数据的simScore
      .filter(_._2._2 > 0.6)    // 过滤出相似度大于0.6的
      .groupByKey()             //把mid相同的a与不同电影的相似度做聚合
      .map{
        //将聚合后的同一电影对不同电影的元组集合转换为推荐列表(按相似度排序)
        //并将列表中的每一个元组( b._1, simScore )转换为Recommendation类型
        case (mid, items) => MovieRecs( mid, items.toList.sortWith(_._2 > _._2).map(x => Recommendation(x._1, x._2)) )
      }
      .toDF()
    movieRecs.write
      .option("uri", mongoConfig.uri)
      .option("collection", MOVIE_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    spark.stop()
  }

  //自己定义的 求向量余弦相似度 的函数
  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix):Double ={
    //两个特征向量的点乘结果 除以 向量的模长乘积
    movie1.dot(movie2) / ( movie1.norm2() * movie2.norm2() )
  }

}
