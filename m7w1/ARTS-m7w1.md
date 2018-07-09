# Algorithm
## [Leetcode 729 Daily Temperatures](https://leetcode.com/problems/daily-temperatures/description/)

* difficulty: Medium
* description: Given a list of daily temperatures, produce a list that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.For example, given the list temperatures = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].

* analysis: 根据题目，可知要找到每一个数字后面第一个比它大的数字，并计算下标距离。首先考虑暴力解法，很简单，当数组过长时肯定会TLE，所以暴力解法不可取。因此想到考虑用栈的思路，栈根据条件必然是降序排列，因此导致站内数字出栈的必定是数字后面第一个比它大的数字。数组从前向后遍历，遇到比栈内数字大的就将出栈即可，但是这样无法记录数字之间的下标差。因此，栈内考虑存储下标即可解决问题。

* code
    ```python
    def dailyTemperaturesWithStack(self, temperatures):
        if temperatures == None or len(temperatures) == 0:
            return []
        length = len(temperatures)
        result = [0] * length
        stack = []
        for i in range(length):
            while len(stack) != 0 and temperatures[i] > temperatures[stack[-1]]:
                index = stack.pop()
                result[index] = i - index
            stack.append(i)
        return result
    ```


# Review
## [The Problem You Solve Is More Important Than The Code You Write](https://levelup.gitconnected.com/the-problem-you-solve-is-more-important-than-the-code-you-write-d0e5493132c6)
这篇文章讨论了一个非常根本的问题，就是我们code是不是解决了一个真实的问题，而不是为了code而code。

文章给出了个观点:

1. Not every code is worth writing
2. Not every bug is worth fixing
3. Not every command is worth scripting
4. Not every feature is worth coding

从这篇文章中可以看出，工作中拿到需求首先要进行思考，而不是上来就写代码。写代码，固然是开发人员的吃饭家伙，但是滥写代码并不值得提倡。首先要思考需求，这些模块是不是一定要code，能不能得到更简单的实现办法；并不是写的越多，工作量越饱满就越好，更应该注重的是工作中的思考的广度和深度。其次是对bug进行优先级划分。第三，脚本这个没太看明白，“You don't need to spend the time to code scripts if you’re going to hide essential knowledge of how the underlying command works.”。最后就是，不是所有功能都需要编码。

作为一个开发人员，如果你明白需求到底是怎么回事，你就有可能写出精简且质量高的代码。写代码的目的应该是满足社会的需要，并且创造价值，而不是满足你以自我为中心的对世界的看法。最后有一句话共勉：“If all you have is a hammer, everything looks like a nail”。

# Tip

最近有个小需求，生产环境上的应用是java的，要上线的新应用是python的。为保证速度和稳定性，不考虑调用http接口，因此将python工程打成jar包发上去。

使用https://github.com/yishenggudou/jythontools：
```
python {path/jytoollib.py} {path/hellojython.py} main
```

# Share
## JWT鉴权

* 什么是JWT？
    
    Json web token (JWT), 是为了在网络应用环境间传递声明而执行的一种基于JSON的开放标准（(RFC 7519)。该token被设计为紧凑且安全的，特别适用于分布式站点的单点登录（SSO）场景。JWT的声明一般被用来在身份提供者和服务提供者间传递被认证的用户身份信息，以便于从资源服务器获取资源，也可以增加一些额外的其它业务逻辑所必须的声明信息，该token也可直接被用于认证，也可被加密。

由于HTTP本身属于无状态协议，所以为了在客户端和服务器保存用户的登录信息，我们需要使用session和cookie来保存用户信息和进行验证。也就是在用户登录之后，服务器端为用户创建了特定的session，用来标识、跟踪用户。在服务端保存Session的方法很多，内存、数据库、文件都有。集群的时候也要考虑Session的转移，在大型的网站，一般会有专门的Session服务器集群，用来保存用户会话，这个时候 Session 信息都是放在内存的，使用一些缓存服务比如Memcached之类的来放 Session。

Session生成后，服务器端则通过cookie来识别用户。每次HTTP请求的时候，客户端都会发送相应的Cookie信息到服务端。实际上大多数的应用都是用 Cookie 来实现Session跟踪的，第一次创建Session的时候，服务端会在HTTP协议中告诉客户端，需要在 Cookie 里面记录一个Session ID，以后每次请求把这个会话ID发送到服务器，我就知道你是谁了。

但是传统session方式在随着用户数据急剧增多，暴露出以下问题：

* Session: 每个用户经过我们的应用认证之后，我们的应用都要在服务端做一次记录，以方便用户下次请求的鉴别，通常而言session都是保存在内存中，而随着认证用户的增多，服务端的开销会明显增大。

* 扩展性: 用户认证之后，服务端做认证记录，如果认证的记录被保存在内存中的话，这意味着用户下次请求还必须要请求在这台服务器上,这样才能拿到授权的资源，这样在分布式的应用上，相应的限制了负载均衡器的能力。这也意味着限制了应用的扩展能力。

* CSRF: 因为是基于cookie来进行用户识别的, cookie如果被截获，用户就会很容易受到跨站请求伪造的攻击。

而JWT鉴权则可以避免在服务器端保存用户认证信息，这就意味着基于token认证机制的应用不需要去考虑用户在哪一台服务器登录了，这就为应用的扩展提供了便利。主要流程如下：

* 用户使用用户名密码来请求服务器
* 服务器进行验证用户的信息
* 服务器通过验证发送给用户一个token
* 客户端存储token，并在每次请求时附送上这个token值
* 服务端验证token值，并返回数据

这个token必须要在每次请求时传递给服务端，它应该保存在请求头里， 另外，服务端要支持CORS(跨来源资源共享)策略，一般我们在服务端这么做就可以了Access-Control-Allow-Origin: *。

### JWT构成
WT是由三段信息构成的，将这三段信息文本用.链接一起就构成了Jwt字符串
```python
    eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ
```
#### 头部
第一部分我们称它为头部（header),第二部分我们称其为载荷（payload)，第三部分是签名（signature)。

jwt的头部承载两部分信息，一是声明，即jwt；二是加密算法，通常直接使用 HMAC SHA256：
```Js
    {
        'typ': 'JWT',
        'alg': 'HS256'
    }
```
然后将头部进行base64加密（该加密是可以对称解密的),构成了第一部分。

#### Payload
载荷就是存放有效信息的地方。这些有效信息包含三个部分：标准中注册的声明，公共的声明，私有的声明。

标准中注册的声明 (建议但不强制使用) ：

* iss: jwt签发者
* sub: jwt所面向的用户
* aud: 接收jwt的一方
* exp: jwt的过期时间，这个过期时间必须要大于签发时间
* nbf: 定义在什么时间之前，该jwt都是不可用的.
* iat: jwt的签发时间
* jti: jwt的唯一身份标识，主要用来作为一次性token,从而回避重放攻击。

公共的声明 ：
公共的声明可以添加任何的信息，一般添加用户的相关信息或其他业务需要的必要信息.但不建议添加敏感信息，因为该部分在客户端可解密.

私有的声明 ：
私有声明是提供者和消费者所共同定义的声明，一般不建议存放敏感信息，因为base64是对称解密的，意味着该部分信息可以归类为明文信息。

#### Signature
签名由三部分组成，header (base64后的)，payload (base64后的)，secret。

这个部分需要base64加密后的header和base64加密后的payload使用.连接组成的字符串，然后通过header中声明的加密方式进行加盐secret组合加密，然后就构成了jwt的第三部分。


### Application
一般是在请求头里加入Authorization，并加上Bearer标注。
```js
fetch('api/user/1', {
  headers: {
    'Authorization': 'Bearer ' + token
  }
})
```
服务器验证流程：

![avatar](jwt-authentication.png)