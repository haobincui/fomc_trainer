# chk3 重生成样本逐条审计清单

生成时间：2026-08-05 12:37:00 UTC  
对应 release：`chk3_minutes_clean_v1_20260805`

## 范围与读法

本清单列出最终 release 中所有 `target_mode=deepseek_regenerated` 的 52 条样本。其余 2,020 条
target 均为 acquisition response 在 clean analysis 上重新通过当前 validator 后复用，不在本表
重复列出。

- `Source recovery response ID` 只适用于 46 条 `deepseek_point_in_time_recovery`；其余 6 条
  analysis 来自确定性 transport/citation projection，因此记为 `—`。
- `Target response ID` 是最终进入 release 的 chk3 Minutes teacher response。
- `Analysis SHA256` 和 `Response SHA256` 取自最终 per-row manifest，可用于将本表与 immutable
  release 逐行核对。
- 所有 response 均由 `deepseek-v4-pro` 返回，fingerprint 为
  `fp_9954b31ca7_prod0820_fp8_kvcache_20260402`。
- validation 样本 `chk1-analysis-2022-11-02-0c0691811be35f30` 的 source recovery response
  先生成了带 `ev-` citation 的完整 analysis，随后执行确定性 citation projection；因此表中
  source response ID 指向投影前 cache，而 `Analysis SHA256` 是投影后的最终训练输入哈希。

## 数量核对

| Split | Point-in-time recovery | Valid transport projection | Evidence citation projection | Target 重生成合计 |
|---|---:|---:|---:|---:|
| train | 36 | 3 | 1 | 40 |
| validation | 4 | 0 | 0 | 4 |
| test | 6 | 0 | 2 | 8 |
| **合计** | **46** | **3** | **3** | **52** |

## 逐条 lineage

| Split | Sample ID | Analysis mode | Source recovery response ID | Target response ID | Analysis SHA256 | Response SHA256 |
|---|---|---|---|---|---|---|
| train | `chk1-analysis-2009-08-12-cf69bf9cf38c8003` | `deepseek_point_in_time_recovery` | `ed537fd2-3b03-4396-8908-6a522542b017` | `654c72c2-7d50-4d08-9c22-2a3af5880035` | `76643bee0cdc23d8487969bf1119409b80cbce7c9ea8d83447e13da922224768` | `983d14fde4d61de73c1b2356be8070197ad915cd722ab767a240bc994a89f99d` |
| train | `chk1-analysis-2010-08-10-b668e691d12967c0` | `deepseek_point_in_time_recovery` | `b576c0af-b616-4567-b883-52ada759fd68` | `f594425a-c37d-4630-bc7b-9faf63a0b28d` | `fb7273c5dcaa1f7f8b82586b170cad8b95310e397d1f9d3c2530a7f66bbf93ca` | `e0e4f237058544ed656ce9b05086f9eef6cd7a9a030e6885f8b0d441af65bf2c` |
| train | `chk1-analysis-2011-11-02-80492c356c1f1744` | `valid_transport_projection` | `—` | `a48b5540-bdf5-450f-9ea9-1a347d712871` | `ea2d77ef8959b56d14248e50003487f00d4acd6e99ae6e4d9b53f02ead0d2fc6` | `d79fc1fd1f371e15a0b1ede24b1764465174f07ba784eb930d5136b21eeec7b2` |
| train | `chk1-analysis-2012-01-25-b4f0422ca9e358cc` | `deepseek_point_in_time_recovery` | `03fb4d6a-6a37-456f-a13a-d34009cd8392` | `3c070255-94d0-497c-bd10-a5ba3950d308` | `b0148259e7e402082cab22e4f289baf1bb80e5db3dfd0f5d977c6e251de095df` | `88ff47426099cc21f03db5aecf534d7a9701984f1d9ee9d33d8a21dd96b63cc0` |
| train | `chk1-analysis-2012-04-25-0ef189369c3cb95d` | `deepseek_point_in_time_recovery` | `03f56104-f413-4bde-b3a3-a8c4a4ad2552` | `fd7164fa-4475-4dcb-b965-455f79eb7c15` | `a9c9c0b68a2ba91ebf1bdadc9b4f2ac53897002aad4a23e587fd5748d7514bc5` | `a7c38ebddea4bf09393cf145972fb228334f53aa33122c6f0a3bcfd87e4efbe0` |
| train | `chk1-analysis-2012-09-13-8a23ba3aa8ac309f` | `deepseek_point_in_time_recovery` | `599f15d4-1a4c-4eec-a76d-f4c6ff82ee60` | `62e651ee-6c7c-4eee-bf4c-a1c45fcfd2d3` | `a32490de79329e9075feb45f862582f8fade8802e2d22f5d5584d2da9e059a7a` | `fbaed745c764afe4b9b909074410e8f75f75183db7fd170f4cfbfc5de1f79d23` |
| train | `chk1-analysis-2013-05-01-23e3adeec1441951` | `valid_transport_projection` | `—` | `ac86b06d-5e38-4118-b738-9a278907ccea` | `2b3b8f5e9e2895f1adfb208d380b3e4f70f238268a95d727c76ff4a79ffb354c` | `02603d51c70ae7e543803acf842593be980c105651e3d4c70ce089f1e57f8c20` |
| train | `chk1-analysis-2013-10-30-eba6e1a404b795dd` | `deepseek_point_in_time_recovery` | `1fa376d8-1a98-4489-a5be-2b8028432dc1` | `dcb58494-bb76-41ef-8415-37c722b0ad6d` | `3ab347244495e8289fe06be2cc059ec4c8384885b4574626ebeda2a667ee30bb` | `1a544c69d22fa3af6716a7b3ac0be0a3ba55522b261cdf834738ad1142ab82be` |
| train | `chk1-analysis-2014-01-29-052ec66c2e3dac80` | `deepseek_point_in_time_recovery` | `da7714e9-2993-48f0-adec-8f30f51d4c43` | `70118ccd-5110-4304-9de2-06f5ba53fc2f` | `009d1a137a54e8bbd351007b6169d045327194fb31074cf3fbb94037c8cb272d` | `2a780d49bb8ffc1b8db3be186c8c1202bf720a82ecab2d963141f061b9c92b44` |
| train | `chk1-analysis-2014-04-30-6464cdc92348378d` | `deepseek_point_in_time_recovery` | `fb360790-4445-4d71-843d-704f46128d44` | `4eab8c9a-1fba-4e26-806b-45dd346cc9f8` | `d1698bcdd5e7d3c342e67283b4a5039cd1359bdac3659166ff8e608d462fd777` | `0c337f7c5d5b17e9753963c0a1a7a1f0c8b0ad804afcef3f1ace0db7fe7b2078` |
| train | `chk1-analysis-2014-06-18-8091fb24a0b93f53` | `deepseek_point_in_time_recovery` | `cb23bdb6-632c-4514-910b-46f47e748a56` | `ac87abf2-e0cd-45ae-9842-0b2bc81847ea` | `53472882524ee02b68c6baa6838bef721b6516de56145b328cf3546d5c38add6` | `731b27c93332a4b00d06ab52fccba2a0bd5600ac2faf026f18491835c56aa972` |
| train | `chk1-analysis-2014-07-30-3fa3896d440bf31d` | `deepseek_point_in_time_recovery` | `e95aec16-efd3-4847-a708-3b57d5aa4fc9` | `101341d0-13a5-44ec-b937-6e530d3ce7fc` | `254d2e539bc831e244e74f12f0f909ea98fb76b569fd65d092054a6a1400e4b5` | `477adf151decaa043eff78697bee0e41e588c92fb775f9673be05fa8a8c0ff38` |
| train | `chk1-analysis-2014-09-17-7cf0fe6216d56cd8` | `deepseek_point_in_time_recovery` | `bd386e6d-f6fc-4555-9c60-b10e845e10b1` | `d2788edc-1275-4e2e-8374-9e6d37e76c94` | `a1d3353cc86289be0b1ffdee6b99269b2d952fdf820dbf430533f39f8abd39ae` | `973b85b81be0fe745785dcc005a9b26b38e2283851b7e69f04a7728ddad1867f` |
| train | `chk1-analysis-2014-09-17-e40615f78d7d3605` | `deepseek_point_in_time_recovery` | `64ed5091-cbf4-4375-ac1d-2777946da36d` | `bb62b3f4-92bd-4283-8894-e495259a8603` | `c9654127e2c6c26bb376e94d990b6bb1259eb4f2dab569508bbd87959a61e480` | `37fd79a7310ccc381e964d56d5a1c760be9eca831c8f645ffc6a4db9d5556de9` |
| train | `chk1-analysis-2015-01-28-130eab2501bba36f` | `deepseek_point_in_time_recovery` | `aaf52465-7488-4062-bd49-b7e66ca5dffd` | `33630888-3f79-4945-95df-bc37120578e4` | `1941cdbe954acdc8d8524efccc6b9cc34c93a06350958afa9b0288acacd1df61` | `6cae46bfbc7461ba4291769278ee0159d1883c2a35d8c95e568f6e2b31ce9367` |
| train | `chk1-analysis-2015-01-28-1b466eed2339cbfa` | `deepseek_point_in_time_recovery` | `e2a518de-45d5-4ad9-a3b6-2ea35037f576` | `98aea466-6309-4dc6-9fa0-1c52e685dd14` | `6ec40ebdda8516648f02eb9acfb2fec0df38ca9d1a76d050b86f522b1bbb01b2` | `a9f95799bf7a9a0cf9f88cf923b04ce3995c9d3eb7449c2930754b437b0dd1c9` |
| train | `chk1-analysis-2015-04-29-9183e17de21116b6` | `deepseek_point_in_time_recovery` | `39c085cc-004b-4a05-8a82-a38ea5803025` | `9bb9659e-2cee-4256-9411-826c80e2be59` | `ebc8e03d00fd07389d46b929340e218445b114a3e27fd2b96463914ea80f8353` | `0670cd03f7da014fd6997299e4340cced736fbd1d234d0df38475dcda665dbf2` |
| train | `chk1-analysis-2015-04-29-b36c68d115100189` | `deepseek_point_in_time_recovery` | `31981340-e92e-4b61-aafc-f8991774547f` | `298ea90b-ea75-45ef-a529-61c9403e796b` | `ad4685a35f7fa198c44daa8a34f506020ab82249f939a6832bb6df179070dee9` | `1575cde24eac8a13abafb63ac70b6fac57668fc440022fcd863e6a595bdbe8d1` |
| train | `chk1-analysis-2015-09-17-b9169430c0bcf51a` | `deepseek_point_in_time_recovery` | `713984e8-d81c-4786-af50-50cc4bf80d58` | `3a7c5a0a-83d7-435f-b5d8-430cf7d57a89` | `6c7c7adfe3e3933f718fad0db04746896e5cb61353c4319affc703740778b050` | `59eba5ca56662d657403e7e331df34dd67c61d5626967a9180d590b27bd95e59` |
| train | `chk1-analysis-2016-04-27-2862ccfc0ebd1b89` | `valid_transport_projection` | `—` | `16ab6a50-4f25-4ef0-95ed-6d96bf7ec99a` | `0b33fee1a03b3c31300ab0687b568c9feceb4fcd533c26b349ba780e47f8d56b` | `1c40f0c49fef8f72a49f2fbbc65d92b58c5bfce79809d6274b85efda99e61d7d` |
| train | `chk1-analysis-2017-03-15-80c00642b761cb75` | `deepseek_point_in_time_recovery` | `64d2a836-5e36-441f-9516-b13ff601e830` | `04f025dc-e8e0-444c-a65c-a54f93d346bc` | `c5cdd16c7b63fb4ed4d59e30ece0660053d3aaa214729896c5ac8b999726113f` | `ab1b57f1f9eb4611eadae236754046d050ffdafbc8eca07b310d3a36dd216b44` |
| train | `chk1-analysis-2017-03-15-fd133c795b21e8b7` | `deepseek_point_in_time_recovery` | `f30aeb2d-9ae1-47b4-ae9a-f3d3dcfffe75` | `05122985-38b3-448e-98a9-e1e16e8ee9c9` | `27103af662f304d95efa56da896571c0af6cc913abfb5e335724116762f73cc3` | `3e156aaf3bc36290ba763d0261c4f9993095754782a8adbd0d4374739e16f535` |
| train | `chk1-analysis-2017-07-26-1504196022fc3fdc` | `deepseek_point_in_time_recovery` | `8f709c84-4156-4eee-b25d-00b549a80db0` | `2f39b9cc-ce3f-426c-9f86-f68ebfd7eba5` | `0ed6afcd8cb26152201f9bf5884e6db9e3d865004df259f68765ceb806a788f9` | `93a3021508dabe5a23fa57197cb00d4c60a7f9caf6ad17d844fafac6f5810a0e` |
| train | `chk1-analysis-2017-07-26-4dd8c646a810e9f8` | `deepseek_point_in_time_recovery` | `44d6785b-356a-4776-8d57-185066c47fa0` | `bc877881-225a-4947-b391-4cafd90f571e` | `4741333ea34d3b6af208a8ffff5536233bfce34f934be91b365e1d11e62b9ded` | `49f433ccb27b25c87876d734d6c2a22e407330f4b7763042b7de4d63638671aa` |
| train | `chk1-analysis-2017-09-20-26b62fdc56fa1199` | `deepseek_point_in_time_recovery` | `94e32505-57d0-4368-8d96-ef7ca6736da0` | `74ae5d77-5884-4af0-83e1-9bbe3014c1ee` | `7a7a188096b684c31c1f66248576c1f908bde0a0895b7e6ffac87970d75ce90a` | `75be1ec1a80a2fe531eefa765561da3c6059afa9304a15597eff911dfc9764ad` |
| train | `chk1-analysis-2017-09-20-50ee8446a4c3150f` | `deepseek_point_in_time_recovery` | `0cbc56af-adb6-40c4-b21a-9a5b20858035` | `cf1f5a0d-1fe6-49fa-adf2-2a6abf385541` | `16b1587a9d38aef3516138e36e23496a076e3158b3a742de410b4f75469f6896` | `8b7b3b2b3b72693093b565589747caeeead65ab9c197cf49e412913b624d54b9` |
| train | `chk1-analysis-2017-11-01-48437d0aef5b69d9` | `deepseek_point_in_time_recovery` | `6e74f229-742d-4f26-bb02-a5911731dfdc` | `1f65e22a-b63b-4d60-b151-e1e8066af11b` | `418a1119737fb5640a60360225451dc1b9134beadad99c003d75ea06d5e475ef` | `adb87e2870cf62717c71ae3599fc9aba93ec41f66268debbeae63ab47e77a060` |
| train | `chk1-analysis-2017-12-13-a24c5b2434c0b9e6` | `deepseek_point_in_time_recovery` | `5e57dbe2-c499-40d1-bd5e-d1fdae70ab86` | `84032297-d1c7-48ce-9ef0-39e0e4401e19` | `be01fce28da94af8288da24d904836794609dbe3a937010b7f4bb72bee9c070a` | `021bee0c2a293d72b91d15207649de51a05212b54cd9fc765eef4a236955fa45` |
| train | `chk1-analysis-2018-01-31-572c4cacb719774c` | `deepseek_point_in_time_recovery` | `b4af99bd-ce27-4f01-8bba-9371acc53317` | `62b3999b-9a9d-4757-930b-f254ea5cdcf2` | `351cc361d890369bc8abb69152692f87fec4fd852dc1b3cec83661a50b445f3e` | `c2c8c74ce93a3f59d3971c2db817504a94230ebb0d1761bb1dd8416feac9847a` |
| train | `chk1-analysis-2018-05-02-1902f0eed290ac05` | `deepseek_point_in_time_recovery` | `9a075673-663c-4ff4-9960-6d5e1f427c61` | `9149f81e-83d3-485f-9d3e-a203d94fd3f2` | `8728f8e4160e8515bdde1e6c9acc26363aeade7c0fc9e73974261ca715c74a60` | `1dee1365fd45dd69be30ccf50c619cb40ba1cfcb84435a287da300a694b9ca41` |
| train | `chk1-analysis-2018-05-02-b27cb240a2a44553` | `deepseek_point_in_time_recovery` | `7968391c-2b22-45ad-8663-375700433004` | `e35a073c-3bff-469d-9c53-76617a675b1f` | `f82bb4d4efd493136c0d89b63bf24135d7da224f45f69d87e343db35659e0380` | `38e2b1be5945c36b051b511c1dc4367b8afee28ae8faf867aa0cb93fd4b387f3` |
| train | `chk1-analysis-2018-06-13-741fefbb2cd05fe9` | `deepseek_point_in_time_recovery` | `475b5e51-954c-4ee0-8c4d-2bd9b6e88670` | `2a36b728-07e6-4e2c-bd10-9a4e3729ded0` | `6efee3571f47e4d35475222e94769038bf1f111efc6a6872347832db2e0200c4` | `83431d975c1716d70ef2de7730766a0bc8b4f277825bb9af3765ffe121e7ebd8` |
| train | `chk1-analysis-2018-09-26-679d1315f1467a4d` | `deepseek_point_in_time_recovery` | `86de9a73-2487-4fb3-8026-ce324cd19db6` | `05ecc2ad-c0e1-49fc-a481-63776a0088cf` | `fe0b005e8288a6aa3ef10fb7dcb50cce5bf8d4f4a3cbdc6a700e1b52fdf64c40` | `d9a29a38ef04b3c614008e81c9e2ac292d35a3e292ebe4de04cad7b8cb7b8d0e` |
| train | `chk1-analysis-2018-12-19-b42ab5d44d4468d5` | `deepseek_point_in_time_recovery` | `564f606a-b114-4fdd-ae68-14e954ff2de0` | `2f835c66-975b-4c95-978d-a6b57c0367d2` | `137f3bff99cbf9d98b7ed6a3dbdf18ab0076ae3a0919345a477a8e20aa0ad5b0` | `a587d8aa47c96c6856f01aba1b52436e08a294e5d394733aff0fd7e05a88e601` |
| train | `chk1-analysis-2019-03-20-04f2301060055574` | `evidence_citation_projection` | `—` | `8570a5b7-8adc-460d-b9c5-c10fcd633b54` | `627e0387d5c7f6a599a981824e8bca226391e5e836a400063bd867aacb5a3a86` | `03ad781a4b886aeaa8082e51763dbe2fcba1da92ed3ec2598019d91eb83934f8` |
| train | `chk1-analysis-2019-06-19-08f729f78295225a` | `deepseek_point_in_time_recovery` | `c7808879-8522-488d-bef8-853d81541209` | `74dea458-1f9e-4b8a-a092-7df9e2a20fba` | `d49c24d7c24539ca70778625778bbe065cf1d8dfb184d66655e52e03a38e4a01` | `b8f624b5097a9613908a0c7671ac9f2e50da7b830623008e8ea7e3baeec853f2` |
| train | `chk1-analysis-2019-07-31-76b1890136824dc6` | `deepseek_point_in_time_recovery` | `a71d3f47-d669-413b-b2d3-65aa2c4217b3` | `c1df44f4-3b94-47f2-a13d-879cc6b46281` | `1dc793655ffbc48d52b09b0cecb9b167213c76994c25899657b8cb386c93a0db` | `8a19316afc706f7e1284cf034edc46c7ac8a13cfd83b5bfb7b3b8ce1d3c9408d` |
| train | `chk1-analysis-2019-12-11-ae46ebf55c965b6d` | `deepseek_point_in_time_recovery` | `829bf911-87c7-46c0-9c9d-7e17f8c0b2d3` | `93a1d374-2334-40a7-bc98-ac947c09dc5a` | `b951a6daf91dcdd2306e55e2c603e4f30bb83e5ca3f34a256541b1974ed523a3` | `b89e660ce60c37747efd95e0cc956bdf3dc989bc4e147058fa9dda2e22862cf2` |
| train | `chk1-analysis-2020-09-16-8673327cfe796954` | `deepseek_point_in_time_recovery` | `042411d2-575b-41c4-af40-ac40d9614a24` | `f094c313-de3a-42c3-a757-afe36040fbd6` | `a896ebc7725dbdc5d729e66212cafe614ca9063ba5ba1b6e330d45baf2271266` | `f4c4ee86af7a50cae6e1f463ab3a4f8e6b34e41b7f21fd587cbc1c3c2258f0c1` |
| train | `chk1-analysis-2021-09-22-11a33682c71cf594` | `deepseek_point_in_time_recovery` | `b14045c1-c475-43e8-91b8-8f990378f23d` | `09f21f31-9af6-4721-b9a5-696a6eaa271e` | `d009b23156f96bc0731a95fd4dbbc4dcf9d256169af2fe009d2130b95358dce7` | `7eaa3e4d7bde6c99bee165fb63afeaf19a783a7e024784ecbd344682f18e9d39` |
| validation | `chk1-analysis-2022-06-15-21ae70b3e24cdcc9` | `deepseek_point_in_time_recovery` | `3432c6ec-47f0-4a9d-96ab-99333253c57d` | `6370633e-2e5c-491b-a10c-beb5f62b5054` | `ed2f15eb43112a5b12b06fb732c1d7ede5896d1877c304488c4b9520fca0f81c` | `d1c553c52b6b7fbc1f4809fb1dcbe306ea6c5c8f7805eb386fe89b8d5136439b` |
| validation | `chk1-analysis-2022-06-15-d956e5b3566caab1` | `deepseek_point_in_time_recovery` | `6c94eeb1-d830-48c4-b610-e66c1565bbba` | `bc92d49e-d336-468e-9fe7-50d06c65611c` | `fe60f0681e8627da9a347b670d592c95342c10f8306dbe73a9b9b53728641851` | `36c3a7e925b794ea4337981f658ef6ab04d9e7e049e62d758c50042975f49b1d` |
| validation | `chk1-analysis-2022-11-02-0c0691811be35f30` | `deepseek_point_in_time_recovery` | `4ff059ce-01d6-40fe-b0bc-ca1d4b41ddab` | `5f25de5f-2046-4ce5-9c2d-1b6fd4a7c68a` | `6c719831cadbda4868faf18a034a751605996afa510af9281a703b7a8527c70a` | `6dfdcc24c92cf3c8a61555797030b47296183704b66ab194a49c347287fe2071` |
| validation | `chk1-analysis-2023-03-22-fc01dec4c2fd1b6d` | `deepseek_point_in_time_recovery` | `9cf2b104-47d7-406a-b601-3cdd2a1f2f7a` | `941c3a48-2290-41ab-9d3a-67e0d81cba9c` | `f58fd3bd0ac2a5a8183e1654c6d730242b58dbadbb1a6c6f5ae214424ea2f065` | `ce666ecf339d334c39501ecafa004a98de4bf9f11222e9e5a81fd98bf6d6f876` |
| test | `chk1-analysis-2023-11-01-ec133eeeab8c219d` | `evidence_citation_projection` | `—` | `fa6be3ef-42e8-418b-8700-37d3462bc983` | `e783777adf542fa7f4b4d47c29defb883df040d7175b93bfc10469e2950ce4dd` | `14da95bd5cd4d60e11bba6a643f72766aa991289d0f3623c40fca2f80d1b62e7` |
| test | `chk1-analysis-2024-01-31-b7941d36e1db346a` | `evidence_citation_projection` | `—` | `884e50f4-abe8-47ae-9d83-a7663fbbbda3` | `09865039e6109c2a47334c2abb29c05e3aff4a4dcc0119fe5211dd69bbfb4f5c` | `c7d8d7e0062c498f94cc90b2cf541cbe7e9930caf999c815c554e9b271cede6b` |
| test | `chk1-analysis-2024-03-20-3be733adee35d977` | `deepseek_point_in_time_recovery` | `b4365563-286f-4a09-baad-3295fa6027af` | `477d8d91-e66a-44fe-af1c-3db65c916798` | `4ce1b59221fb156ffae992dba8055598e46c4745f1c1eac1cf3150f32a539f14` | `bbaf368ca073abcd47bd0ec9d608f17e4e9aab9ccb26017cbcb5a8144f618c1a` |
| test | `chk1-analysis-2024-05-01-311efe26a0415d3f` | `deepseek_point_in_time_recovery` | `efaf2056-4bb4-4f3a-9ff2-0034455a89b0` | `b9ab8ecd-1e06-46d9-a5fe-6e9d7222fca4` | `0246c5e3cf574e29a9ccfd2d033b89b8727441f35d1a515bf59ef70e15719b5a` | `400e937d1f5aebc18160d9ebd549830919e6390231b59fe42012b2bc205f3b99` |
| test | `chk1-analysis-2024-06-12-b93f418ef45976f0` | `deepseek_point_in_time_recovery` | `2920995d-d453-43a8-92f9-197455c66c93` | `3e14e5c4-104c-44a6-bb87-0768345f1715` | `f427be133c355cd5086fca99dc44dabf3651a02d7c2b7a1a66bff68538b8137f` | `29475b91a63183e9cb06efda11163e0becfcf9e5bd1612ee480b657845ba91e8` |
| test | `chk1-analysis-2024-06-12-e8eba6972f94a00d` | `deepseek_point_in_time_recovery` | `567af01b-175f-483a-9a4a-3b86b7deb477` | `6c7c6a87-e99f-4031-b90b-56fb9d3b1a92` | `c52568ebb86b6e89151666b56240de5b18e4e935fd3d4fafed232fbea98d2e43` | `6b9c4aa9943badde9d657d8aeb7553c1d8402e0ba7bb077cfe00ca7cfc41657a` |
| test | `chk1-analysis-2024-11-07-89e6928dc6977a55` | `deepseek_point_in_time_recovery` | `7f8235d9-48f0-488d-be7b-52bd040caa60` | `aa6cf737-0ba3-4d62-bf54-4d600b79d383` | `e6a82ee6cf0707e3e11cd8e605dc3c06ba2a0e649552f5fd984b37f5d32ab81e` | `a8df13702c2f5f11ec440b7fa6e981422458d57eb69d086491a525eb75573b8b` |
| test | `chk1-analysis-2024-12-18-d4b86c8b4cc9e06d` | `deepseek_point_in_time_recovery` | `4896c498-6a10-4e21-b135-2bb554553736` | `9d9ed586-50c3-420d-98c7-bfa2be5c4028` | `1369c9823471ba1e9a7ee652dfb360d36fb494d228b75ca61ccf23d618562ff5` | `c6b08db7041eef82314b7c0cc05f0b5342de27948b00b27fe47b48bfdab5788c` |

## 可复核来源

逐行最终状态以以下 immutable manifest 为准：

```text
dataset/processed/retrain_v2/chk3_minutes_clean_v1_20260805/
  minutes_alignment/manifests/{train,validation,test}.jsonl
```

Source recovery 和 target regeneration 的 accepted provider raw 保存在：

```text
output/data/retrain_v2/chk3/training_release_clean_v1/cache/
  analysis/
  target/
```

本清单不替代 release manifest；它是便于人工检查和追溯的派生视图。若二者不一致，应以
immutable release 的 handoff、release manifest 和 per-row manifest 为准。
