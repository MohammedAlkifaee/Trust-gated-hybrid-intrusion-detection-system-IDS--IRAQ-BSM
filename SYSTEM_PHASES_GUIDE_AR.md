# دليل شرح نظام كشف الهجمات في VANET من البنية التحتية إلى الكشف الحي

> أرقام الأسطر في هذا الدليل حسب النسخة الحالية من الملفات داخل المشروع. إذا تم تعديل الكود لاحقاً فقد تتحرك الأرقام قليلاً، لذلك استخدم اسم الدالة مع رقم السطر كمرجع أثناء المناقشة.

## 1. الجملة العامة للنظام

النظام هو خط كشف هجمات VANET يعمل على رسائل BSM القادمة من F2MD/SUMO. يبدأ من تشغيل السيناريو واستخراج الرسائل، ثم إضافة نوع الهجوم، ثم تدريب أو تحميل نماذج RSU متعددة الرؤوس، ثم تشغيل الكشف دون اتصال أو بشكل حي. القرار النهائي لا يعتمد على نموذج واحد فقط، بل يعتمد على:

1. خصائص الحركة والتوقيت من رسائل BSM.
2. أعلام فحص OBU للقيود الفيزيائية والبروتوكولية.
3. رؤوس RSU المتخصصة حسب عائلة الهجوم.
4. دمج احتمالات الرؤوس في `p_final`.
5. درجة ثقة لكل مركبة `trust_score`.
6. قرار نهائي `final_decision`.

## 2. خريطة الملفات

| الملف | الدور | أهم المواقع |
|---|---|---|
| `vanet_ids_v2.py` | واجهة الأوامر وتشغيل النظام كاملاً. | `main` سطر 2001، `interactive_menu` سطر 1772، `print_cli_home` سطر 1324. |
| `vanet_ids_rsu_core.py` | القلب الفني: قراءة BSM، هندسة الخصائص، التدريب، النماذج، الدمج، الثقة، والكشف. | `RSUMultiHeadTrainer` سطر 1192، `RSUMultiHeadRuntime` سطر 1635، `ArchivedFamilyEnsembleRuntime` سطر 2638. |
| `rsu_trainer_all_in_one_v7.py` | واجهة تدريب النماذج من GUI أو CLI، وتستدعي `RSUMultiHeadTrainer`. | `run_training_cli` سطر 15، `launch_gui` سطر 62. |
| `add_attack_id.py` | إضافة `attack_id` إلى CSV بعد الاستخراج. | يستدعى من `add_attack_id_to_csv` في `vanet_ids_v2.py:394`. |
| `results and models/models` | مجلد النماذج متعددة العائلات التي بنيت سابقاً. | يستخدمه `DEFAULT_DETECT_MODELS_DIR` ويقرأه `latest_archived_family_bundles`. |

## 3. البنية التحتية والتشغيل

| الجزء | مكانه في الكود | وظيفته |
|---|---|---|
| مجلد F2MD | `vanet_ids_v2.py:54` | المسار الافتراضي `/home/instantf2md/F2MD`. |
| مجلد النتائج | `vanet_ids_v2.py:55` | المسار الافتراضي `/home/instantf2md/F2MD/f2md-results`. |
| سكربت الاستخراج | `vanet_ids_v2.py:56` | يشير إلى `extract1_intermsg.py`. |
| لوحة live IDS | `vanet_ids_v2.py:58` | تفتح Dashboard خارجي للمراقبة. |
| ملف التدريب | `vanet_ids_v2.py:59` | يشير إلى `rsu_trainer_all_in_one_v7.py`. |
| مجلد النماذج الحديث | `vanet_ids_v2.py:63` | `release_v3`. |
| مجلد النماذج المؤرشفة | `vanet_ids_v2.py:64` | `results and models/models`. |
| مجلد نماذج الكشف الافتراضي | `vanet_ids_v2.py:65` | يفضل النماذج المؤرشفة إذا كانت موجودة. |
| أوامر الخدمات | `vanet_ids_v2.py:167` | بناء أوامر `start-sumo`, `start-scenario`, `start-live-ids`. |
| فتح Terminal جديد | `vanet_ids_v2.py:284` و `348` | تشغيل الخدمات في نوافذ منفصلة. |

القائمة الحالية في البرنامج:

| الرقم | الأمر |
|---|---|
| 1 | `status` |
| 2 | `start-sumo` |
| 3 | `start-scenario` |
| 4 | `start-live-ids` |
| 5 | `list-models` |
| 6 | `trainer-gui` |
| 7 | `extract` |
| 8 | `label` |
| 9 | `train-rsu` |
| 10 | `detect-offline` |
| 11 | `detect-live` |
| 12 | `verify` |
| 13 | `performance-score` |
| 14 | `edit-config` |
| 15 | `help` |
| 16 | `exit` |

## 4. تسلسل العمل الكامل

| الترتيب | المرحلة | الأمر | الدالة |
|---|---|---|---|
| 1 | تشغيل SUMO/TraCI | `start-sumo` | `cmd_launch_daemon` و `open_in_new_terminal`. |
| 2 | تشغيل سيناريو F2MD | `start-scenario` | `cmd_run_scenario`. |
| 3 | تشغيل Dashboard | `start-live-ids` | `run_live_ids_dashboard` سطر 419. |
| 4 | استخراج BSM إلى CSV | `extract` | `run_extract_intermsg` سطر 374. |
| 5 | إضافة نوع الهجوم | `label` | `add_attack_id_to_csv` سطر 394. |
| 6 | تدريب النماذج عند الحاجة | `train-rsu` أو `trainer-gui` | `run_train_rsu` سطر 1482، أو `rsu_trainer_all_in_one_v7.py`. |
| 7 | كشف offline | `detect-offline` | `run_detect_offline` سطر 1368. |
| 8 | كشف live | `detect-live` | `run_detect_live` سطر 1426. |
| 9 | تحقق من النتائج | `verify` | `run_verify` سطر 1550. |
| 10 | عرض الأداء التاريخي والحالي | `performance-score` | `performance_score_cli` سطر 1146. |

## 5. الهجمات والعائلات

| العائلة | أرقام الهجمات | المعنى العملي |
|---|---|---|
| `pos_speed` | 1 إلى 9 | هجمات الموقع والسرعة مثل ConstPos و RandomSpeed. |
| `replay_stale` | 11، 12، 17 | إعادة إرسال أو رسائل قديمة أو Replay مع Sybil. |
| `dos` | 13، 14، 15، 18، 19 | هجمات حجب الخدمة وتفرعاتها. |
| `sybil` | 16، 17، 18، 19 | تعدد الهويات أو الهجمات المركبة مع Sybil. |

موقع التعريف في الكود:

| العنصر | السطر |
|---|---|
| `ATTACK_TYPES` | `vanet_ids_rsu_core.py:105` |
| `ATTACK_FAMILIES` | `vanet_ids_rsu_core.py:130` |
| `TRAIN_FAMILY_CHOICES` | `vanet_ids_rsu_core.py:136` |
| `ARCHIVED_FAMILY_ORDER` | `vanet_ids_rsu_core.py:137` |

## 6. الثوابت المهمة في النظام

| الثابت | القيمة | مكانه |
|---|---|---|
| `ARTIFACT_FAMILY` | `rsu_multi_head_v3` | `vanet_ids_rsu_core.py:57` |
| `ARTIFACT_VERSION` | `3.0.0` | `vanet_ids_rsu_core.py:58` |
| `ARCHIVED_FAMILY_ENSEMBLE_ARTIFACT` | `archived_family_ensemble_v1` | `vanet_ids_rsu_core.py:60` |
| `DEFAULT_WINDOW_SIZE` | 15 | `vanet_ids_rsu_core.py:62` |
| `DEFAULT_SEQ_LEN` | 20 | `vanet_ids_rsu_core.py:63` |
| `HEAD_VECTOR_ORDER` | ترتيب رؤوس الكشف | `vanet_ids_rsu_core.py:64` |
| `OBU_FLAG_COLUMNS` | أعلام OBU | `vanet_ids_rsu_core.py:73` |
| `DEFAULT_OBU_THRESHOLDS` | حدود OBU الفيزيائية | `vanet_ids_rsu_core.py:83` |

رؤوس الكشف بالترتيب:

| الرأس | الغرض |
|---|---|
| `general` | كشف عام: طبيعي أو هجوم. |
| `pos_speed` | هجمات الموقع والسرعة. |
| `replay_stale` | إعادة الإرسال والرسائل القديمة. |
| `dos` | DoS بنموذج تصنيف. |
| `dos_iforest` | DoS بنموذج شذوذ IsolationForest. |
| `sybil` | هجمات Sybil. |
| `integrity` | سلوك غير متسق أو مخالف لقواعد OBU. |

## 7. مرحلة استخراج البيانات

| العنصر | المكان | التفاصيل |
|---|---|---|
| استدعاء الاستخراج | `vanet_ids_v2.py:374` | `run_extract_intermsg` يشغل `extract1_intermsg.py`. |
| إدخال الاستخراج | مجلد BSM | مثل `/home/instantf2md/F2MD/f2md-results/LuSTNanoScenario-ITSG5`. |
| خرج الاستخراج | CSV | مثل `features_intermessage_v2.csv`. |
| نسخة BSM | `v2` | تدخل كمعامل للسكربت الخارجي. |

شرحها أمام اللجنة:

> هذه المرحلة تحول ملفات رسائل BSM من المحاكاة إلى جدول CSV، حتى أتعامل معها كصفوف منظمة قابلة للمعالجة والتصنيف.

## 8. مرحلة إضافة `attack_id`

| العنصر | المكان | التفاصيل |
|---|---|---|
| الدالة | `vanet_ids_v2.py:394` | `add_attack_id_to_csv`. |
| السكربت المستدعى | `add_attack_id.py` | يضيف رقم الهجوم للصفوف التي تحمل `label=1`. |
| الطبيعي | `attack_id=0` | Genuine. |
| الهجوم | `attack_id=1..19` | حسب نوع الهجوم المختار. |

شرحها:

> بعد الاستخراج أعرف النظام بنوع الهجوم في السيناريو. هذا لا يستخدم لتحسين القرار أثناء الكشف، بل يستخدم لاحقاً لحساب الأداء الصحيح حسب العائلة.

## 9. قراءة BSM الخام من مجلد

إذا كان الإدخال مجلد raw وليس CSV:

| الدالة | السطر | عملها |
|---|---|---|
| `parse_f2md_bsm_file` | `vanet_ids_rsu_core.py:302` | يقرأ ملف `.bsm` ويستخرج المستقبل، المرسل، الزمن، الموقع، السرعة، التسارع، الاتجاه، والثقة. |
| `load_raw_bsm_directory` | `vanet_ids_rsu_core.py:365` | يقرأ كل ملفات `.bsm` داخل المجلد ويحولها إلى DataFrame. |

## 10. التطبيع Normalization

| الدالة | السطر | العمل |
|---|---|---|
| `sanitize_columns` | `vanet_ids_rsu_core.py:170` | تنظيف أسماء الأعمدة. |
| `sanitize_entity_values` | `vanet_ids_rsu_core.py:182` | تنظيف قيم معرفات المركبات والمستقبلات. |
| `_stable_row_key` | `vanet_ids_rsu_core.py:374` | إنشاء مفتاح ثابت لكل صف. |
| `normalize_bsm_dataframe` | `vanet_ids_rsu_core.py:389` | توحيد الأعمدة والقيم والأنواع. |

تفاصيل `normalize_bsm_dataframe`:

| الجزء | الأسطر | التفاصيل |
|---|---|---|
| `row_id` و `row_key` | 393 إلى 395 | إضافة معرف صف ومفتاح ثابت إذا لم يكونا موجودين. |
| الزمن `t_curr` | 397 إلى 404 | استخدام `creation_time` أو `time` أو ترتيب الصفوف. |
| الموقع | 406 إلى 409 | تحويل `x/y` إلى `x_curr/y_curr` إذا لزم. |
| السرعة | 411 إلى 417 | حساب `speed_curr` من `vx/vy` إذا لم تكن موجودة. |
| التسارع | 419 إلى 425 | حساب `acc_curr` من `ax/ay` إذا لم يكن موجوداً. |
| الاتجاه | 427 إلى 430 | تجهيز `heading_curr`. |
| الثقة في القياسات | 432 إلى 462 | تجهيز `pos_conf`, `spd_conf`, `acc_conf`, `head_conf`. |
| معرفات المركبات | 464 إلى 467 | ضمان وجود `sender_pseudo` و `receiver_pseudo`. |
| الملصقات | 469 إلى 483 | تجهيز `attack_id` و `label`. |
| الترتيب الزمني | 485 إلى 487 | ترتيب حسب المرسل والزمن والصف. |
| القيم السابقة | 489 إلى 500 | إنشاء `t_prev`, `x_prev`, `speed_prev`, وغيرها. |
| الفروقات | 502 إلى 517 | حساب `dt`, `dx`, `dy`, `dist`, `dv`, `dacc`, `dtheta`. |

شرحها:

> الهدف من التطبيع أن تصبح بيانات CSV أو BSM الخام بنفس الشكل دائماً قبل دخولها إلى هندسة الخصائص أو النماذج.

## 11. هندسة الخصائص Feature Engineering

الدالة الأساسية: `feature_engineering` في `vanet_ids_rsu_core.py:530`.

| مجموعة الخصائص | الأسطر | ماذا تحسب |
|---|---|---|
| الحركة الأساسية | 535 إلى 550 | `dt`, `dv`, `acc_curr`, `dacc`, `jerk`. |
| الاتجاه | 552 إلى 555 | `dtheta`, `heading_rate`. |
| الإزاحة | 557 إلى 559 | `dx`, `dy`, `dist`. |
| توقع الموقع | 561 إلى 567 | مقارنة الموقع المتوقع بالموقع الحقيقي. |
| زاوية الانحراف | 568 إلى 583 | `dr_angle`, `sin_a`, `cos_a`, وتباين الاتجاه داخل نافذة. |
| أعلام السرعة والتسارع | 585 إلى 588 | `neg_acc_flag`, `low_speed_flag`, ونسبها داخل نافذة. |
| معدل الرسائل | 590 إلى 599 | `rate_msgs_per_s`. |
| تجميع نافذة زمنية | 601 إلى 606 | `mean`, `std`, `max` لكل من `dv`, `jerk`, `heading_rate`, `dist`, `dt`. |
| jitter | 607 إلى 613 | تذبذب `dt` داخل النافذة. |
| freeze ratios | 614 إلى 616 | ثبات السرعة أو المسافة أو الاتجاه. |
| consistency | 618 إلى 625 | الفرق بين المسافة الفعلية والمسافة المتوقعة من السرعة والزمن. |
| تكرار الحالة | 627 إلى 653 | `state_hash`, `state_code`, `state_dup_ratio_w`. |
| شذوذ الزمن | 655 إلى 666 | `dt_z`, `dt_cv_w`. |
| تغير معدل الرسائل | 668 إلى 684 | `rate_ewma`, `rate_cusum_pos`, `rate_cusum_neg`. |

شرح التجميع Aggregation هنا:

> أول نوع من التجميع هو تجميع زمني لكل مركبة داخل نافذة `window_size=15`. بدلاً من الحكم من رسالة واحدة، يحسب النظام المتوسط والانحراف والقيم العظمى ونسب التكرار خلال آخر رسائل للمركبة. هذا يجعل الكشف أكثر استقراراً.

## 12. خصائص Sybil

الدالة: `add_sybil_features` في `vanet_ids_rsu_core.py:691`.

| الخاصية | الأسطر | معناها |
|---|---|---|
| `window_id` | 693 | تقسيم الزمن إلى نوافذ 5 ثوان. |
| `sybil_unique_ids_5s` | 696 إلى 697 | عدد الهويات الفريدة في كل نافذة. |
| `sybil_sender_entropy_5s` | 699 إلى 704 | Entropy لتوزيع المرسلين. |
| `sybil_jaccard_ids_5s` | 706 إلى 712 | تشابه الهويات بين نافذتين متتاليتين. |
| `sybil_new_ids_rate` | 714 إلى 717 | معدل ظهور هويات جديدة. |
| `sybil_new_ids_burst` | 719 إلى 722 | انفجار ظهور الهويات مقارنة بالـ EWMA. |

شرحها:

> Sybil لا يظهر فقط من سرعة أو موقع، بل من سلوك الهوية. لذلك أجمع عدد الهويات وتغيرها داخل نافذة 5 ثوان.

## 13. أعلام OBU

الدالة: `add_obu_evidence_flags` في `vanet_ids_rsu_core.py:731`.

| العلم | مصدره | المعنى |
|---|---|---|
| `flag_speed_phys` | السرعة | سرعة أعلى من الحد الفيزيائي. |
| `flag_acc_phys` | التسارع | تسارع غير منطقي. |
| `flag_hr_phys` | تغير الاتجاه | معدل دوران غير طبيعي. |
| `flag_consistency` | الاتساق | المسافة لا تتوافق مع السرعة والزمن. |
| `flag_proto_nan` | البروتوكول | قيم مفقودة أو غير صالحة. |
| `flag_dt_nonpos` | الزمن | زمن غير موجب. |

الحدود الافتراضية في `DEFAULT_OBU_THRESHOLDS`:

| الحد | القيمة |
|---|---|
| `speed_abs_max` | 80.0 |
| `acc_abs_max` | 12.0 |
| `heading_rate_abs_max` | 2.0 |
| `consistency_err_max` | 5.0 |
| `dt_min` | 0.0 |
| `dt_max` | 2.0 |

الأعمدة النهائية:

| العمود | السطر | معناه |
|---|---|---|
| `proto_anom_count` | 755 | عدد مشاكل البروتوكول. |
| `obu_flag_count` | 756 | عدد أعلام OBU المرفوعة. |
| `obu_risk` | 757 | نسبة الخطر من أعلام OBU. |
| `obu_anom` | 758 | 1 إذا وجد أي علم. |

## 14. تجهيز البيانات قبل التدريب أو الكشف

الدالة: `prepare_thesis_dataframe` في `vanet_ids_rsu_core.py:762`.

تسلسلها:

| الترتيب | السطر | العملية |
|---|---|---|
| 1 | 768 | استدعاء `feature_engineering`. |
| 2 | 769 إلى 771 | تحميل حدود OBU. |
| 3 | 771 | إضافة أعلام OBU. |
| 4 | 772 | إضافة خصائص Sybil. |
| 5 | 773 إلى 774 | تنظيف `inf` و `NaN`. |

شرحها:

> هذه الدالة هي نقطة دخول موحدة. أي تدريب أو كشف يجب أن يمر بها حتى تكون الخصائص والـ OBU و Sybil مطبقة بنفس الطريقة.

## 15. التدريب في Colab أو الجهاز المحلي

التدريب العملي يتم عبر `rsu_trainer_all_in_one_v7.py` أو مباشرة عبر `RSUMultiHeadTrainer`.

| الطريقة | الملف | الدالة |
|---|---|---|
| CLI | `rsu_trainer_all_in_one_v7.py` | `run_training_cli` سطر 15. |
| GUI | `rsu_trainer_all_in_one_v7.py` | `launch_gui` سطر 62. |
| Worker للـ GUI | `rsu_trainer_all_in_one_v7.py` | `TrainWorker` سطر 81. |
| نافذة التدريب | `rsu_trainer_all_in_one_v7.py` | `MainWindow` سطر 110. |
| بدء التدريب من GUI | `rsu_trainer_all_in_one_v7.py` | `start_training` سطر 182. |
| التدريب الحقيقي | `vanet_ids_rsu_core.py` | `RSUMultiHeadTrainer` سطر 1192. |

أمر تدريب مناسب في Colab:

```bash
python rsu_trainer_all_in_one_v7.py \
  --csv /content/features_labeled.csv \
  --out-dir /content/release_v3 \
  --train-family all \
  --window-size 15 \
  --seq-len 20
```

خطوات Colab العملية:

| الخطوة | ماذا تفعل |
|---|---|
| 1 | رفع ملفات المشروع أو ربط Google Drive. |
| 2 | رفع CSV يحتوي `label` و `attack_id`. |
| 3 | تثبيت المكتبات: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `tensorflow`, `joblib`. |
| 4 | تشغيل أمر التدريب أعلاه. |
| 5 | حفظ مجلد `release_v3` أو مجلد العائلة الناتج. |
| 6 | نقل ملفات النماذج إلى الجهاز داخل `results and models/models` أو `release_v3`. |

شرحها:

> استخدمت Colab لأن تدريب كل الرؤوس، خصوصاً LSTM الخاص بـ replay، يحتاج وقتاً وموارد أكثر من التشغيل العادي. لكن نفس منطق التدريب محفوظ في الكود داخل `RSUMultiHeadTrainer`.

## 16. تقسيم البيانات في التدريب

الدالة: `split_sender_disjoint_train_val_test` في `vanet_ids_rsu_core.py:1012`.

| الجزء | السطر | التفاصيل |
|---|---|---|
| فحص المجموعات | 1000 | `_assert_group_support_for_split` يتأكد من وجود مرسلين كافيين. |
| تقسيم test | 1021 إلى 1022 | `StratifiedGroupKFold(n_splits=10)` يعطي تقريباً 10% test. |
| تقسيم validation | 1027 إلى 1031 | `StratifiedGroupKFold(n_splits=9)` يعطي validation من trainval. |
| منع التسرب | 1033 إلى 1037 | يتأكد أن نفس `sender_pseudo` لا يظهر في train وval وtest معاً. |

شرحها:

> التقسيم ليس عشوائياً على مستوى الصفوف فقط؛ هو sender-disjoint. أي أن المركبة التي ظهرت في التدريب لا تظهر في الاختبار. هذا يمنع تسرب هوية المركبة ويجعل الأداء أقرب للتقييم الحقيقي.

## 17. اختيار الخصائص والرؤوس

| الدالة | السطر | الغرض |
|---|---|---|
| `select_general_features` | 785 | اختيار كل الأعمدة الرقمية عدا أعمدة الهوية والملصقات. |
| `select_pos_speed_features` | 790 | خصائص الموقع والسرعة والاتجاه والثقة. |
| `select_dos_features` | 801 | خصائص معدل الرسائل والزمن والتكرار. |
| `select_sybil_features` | 815 | خصائص Sybil ونافذة الزمن. |
| `select_integrity_features` | 820 | أعلام OBU والاتساق. |
| `build_feature_config` | 829 | بناء ملف الخصائص النهائي لكل رأس. |
| `align_feature_matrix` | 860 | ترتيب الأعمدة وملء القيم الناقصة بنفس شكل التدريب. |
| `fit_feature_scaler` | 234 | تدريب `StandardScaler` على الخصائص المستمرة فقط. |
| `transform_feature_matrix` | 243 | تطبيق `StandardScaler` أثناء التدريب والكشف. |

شرحها:

> كل رأس لا يستخدم بالضرورة نفس الخصائص. رأس DoS يهتم بمعدل الرسائل والزمن، ورأس Sybil يهتم بالهويات، ورأس pos_speed يهتم بالحركة والموقع.

## 18. تدريب الرؤوس Head Models

الفئة المسؤولة: `RSUMultiHeadTrainer` في `vanet_ids_rsu_core.py:1192`.

| الرأس | الدالة/الأسطر | النموذج | الهدف |
|---|---|---|---|
| `general` | `_general_estimator` سطر 1209 | LightGBM binary | `label`: طبيعي أو هجوم. |
| `pos_speed` | `_family_estimator` سطر 1225 | LightGBM binary | هل `attack_id` ضمن عائلة pos_speed. |
| `replay_stale` | `fit_replay_lstm` سطر 933 | LSTM | تسلسل رسائل لكل مرسل لاكتشاف replay/stale. |
| `dos` | `_family_estimator` سطر 1225 | LightGBM binary | هل `attack_id` ضمن عائلة DoS. |
| `dos_iforest` | داخل `_fit_single_head_artifacts` سطر 1389 | IsolationForest | شذوذ DoS على عينات DoS-negative. |
| `sybil` | `_family_estimator` سطر 1225 | LightGBM binary | هل `attack_id` ضمن عائلة Sybil. |
| `integrity` | `_integrity_estimator` سطر 1241 | LightGBM binary | سلوك مخالف أو غير متسق. |

معاملات مهمة:

| النموذج | أهم الإعدادات |
|---|---|
| General LightGBM | `n_estimators=500`, `learning_rate=0.04`, `num_leaves=160`, `class_weight=balanced`. |
| Family LightGBM | `n_estimators=400`, `learning_rate=0.05`, `num_leaves=128`, `class_weight=balanced`. |
| Integrity LightGBM | `n_estimators=300`, `learning_rate=0.05`, `num_leaves=96`, `class_weight=balanced`. |
| DoS IsolationForest | `n_estimators=300`, `contamination=0.02`, `random_state=42`. |
| Replay LSTM | `seq_len=20`, `EarlyStopping`, `batch_size=128`, `epochs=20`. |

## 19. Calibration

الدالة: `fit_base_and_calibrator` في `vanet_ids_rsu_core.py:874`.

| العنصر | التفاصيل |
|---|---|
| النموذج الأساسي | يدرّب estimator الأصلي أولاً. |
| المعايرة | `CalibratedClassifierCV(method="isotonic")`. |
| عدد folds | حتى 3 حسب أقل عدد من العينات في الفئات. |
| لماذا؟ | لتحويل مخرجات النموذج إلى احتمالات أكثر استقراراً قبل الدمج. |

شرحها:

> قبل الدمج لا أستخدم قرار كل رأس مباشرة، بل أستخدم احتمالاً معايراً، لأن Logistic Regression أو الدمج يحتاج أرقام احتمالية قابلة للمقارنة.

## 20. تدريب LSTM لرأس Replay/Stale

| الدالة | السطر | العمل |
|---|---|---|
| `make_sequences_per_sender` | 900 | يبني تسلسلات بطول `seq_len` لكل مرسل. |
| `fit_replay_lstm` | 933 | يدرب LSTM ويولد OOF scores إن أمكن. |
| `predict_replay_lstm` | 982 | يحول التسلسلات إلى احتمالات لكل صف. |

شرحها:

> replay/stale يعتمد على ترتيب الرسائل عبر الزمن، لذلك أستخدم تسلسلات لكل مركبة وليس صفاً منفرداً فقط.

## 21. Logistic Regression والدمج Stacking

الدمج الرسمي في حزمة `rsu_multi_head_v3` يتم بـ Logistic Regression.

| العنصر | المكان | التفاصيل |
|---|---|---|
| بناء مصفوفة الدمج | `build_meta_matrix` سطر 1046 | يرتب احتمالات الرؤوس حسب `HEAD_VECTOR_ORDER`. |
| OOF لكل رأس | `_oof_scores_for_head` سطر 1461 | تدريب 5-fold sender-group لتوليد احتمالات خارج التدريب. |
| تدريب Logistic Regression | `_fit_meta_classifier` سطر 1492 | `LogisticRegression(class_weight="balanced", max_iter=400, random_state=42, C=1.0)`. |
| الهدف | `_fit_meta_classifier` سطر 1501 | الهدف هو `label` العام. |

مصفوفة الإدخال إلى Logistic Regression:

| العمود | يمثل |
|---|---|
| `p_general` | احتمال الهجوم العام. |
| `p_pos_speed` | احتمال عائلة الموقع/السرعة. |
| `p_replay_stale` | احتمال replay/stale. |
| `p_dos` | احتمال DoS. |
| `p_dos_iforest` | درجة شذوذ DoS. |
| `p_sybil` | احتمال Sybil. |
| `p_integrity` | احتمال/خطر التكامل. |

شرح Logistic Regression:

> Logistic Regression هنا ليس نموذجاً بديلاً عن الرؤوس، بل هو meta-classifier. يأخذ احتمالات الرؤوس كمدخلات ويتعلم كيف يوازن بينها لإنتاج `p_final`.

## 22. تدريب الحزمة النهائية

الدالة: `fit` في `RSUMultiHeadTrainer`، السطر 1504.

| الخطوة | الأسطر | العملية |
|---|---|---|
| فحص `label` | 1505 إلى 1506 | لا يبدأ التدريب بدون labels. |
| تجهيز البيانات | 1508 إلى 1512 | `prepare_thesis_dataframe`. |
| تقسيم sender-disjoint | 1516 إلى 1521 | train/val/test. |
| بناء خصائص التدريب | 1523 إلى 1527 | `build_feature_config`. |
| تدريب meta على train | 1529 | OOF + Logistic Regression. |
| تدريب الرؤوس على train | 1530 | الرؤوس الفردية. |
| اختيار threshold | 1541 إلى 1544 | أفضل F1 على validation. |
| إعداد الثقة | 1546 إلى 1556 | `base_threshold`, `sensitivity`, `floor`, `ceil`. |
| تدريب نهائي train+val | 1558 إلى 1566 | إعادة تدريب الرؤوس والـ meta. |
| رفض الحزمة الناقصة | 1568 إلى 1569 | إذا `train_family=all` والحزمة ناقصة. |
| اختبار نهائي | 1571 إلى 1585 | `runtime.score_dataframe(test_df)`. |
| تقرير التدريب | 1587 إلى 1614 | Confusion matrix, ROC-AUC, reports. |
| حفظ الحزمة | 1616 إلى 1619 | `save_release_bundle` و `training_report.json`. |

## 23. ملفات الحزمة النهائية

الدالة: `build_required_bundle_files` في `vanet_ids_rsu_core.py:1073`.

| الملف | الوظيفة |
|---|---|
| `manifest.json` | وصف الحزمة والتحقق من اكتمالها. |
| `features.json` | قائمة الخصائص وترتيبها لكل رأس. |
| `scaler.joblib` | StandardScaler للخصائص المستمرة. |
| `general_head.joblib` | رأس الكشف العام. |
| `general_calibrator.joblib` | معايرة الرأس العام. |
| `pos_speed_head.joblib` | رأس الموقع/السرعة. |
| `pos_speed_calibrator.joblib` | معايرة رأس الموقع/السرعة. |
| `dos_head.joblib` | رأس DoS. |
| `dos_calibrator.joblib` | معايرة DoS. |
| `dos_iforest.joblib` | IsolationForest لشذوذ DoS. |
| `sybil_head.joblib` | رأس Sybil. |
| `sybil_calibrator.joblib` | معايرة Sybil. |
| `integrity_head.joblib` | رأس التكامل. |
| `integrity_calibrator.joblib` | معايرة التكامل. |
| `replay_lstm.keras` | نموذج LSTM. |
| `replay_config.json` | إعدادات LSTM. |
| `meta_classifier.joblib` | Logistic Regression للدمج. |
| `trust_config.json` | إعدادات الثقة والعتبة. |
| `obu_thresholds.json` | حدود OBU. |
| `training_report.json` | تقرير التدريب. |

حفظ الحزمة يتم في `save_release_bundle` من `vanet_ids_rsu_core.py:1775`.

## 24. النماذج المؤرشفة التي تم تدريبها في Colab

المجلد المستخدم حالياً:

```text
/home/instantf2md/Desktop/VANET_project/results and models/models
```

بنية كل عائلة:

```text
family/timestamp/
  preproc.joblib
  bin_calib.joblib
  head_pos_speed.joblib  # موجود في pos_speed فقط عندما يوجد رأس صريح
  meta.joblib
  model_meta.json
```

أمثلة من النماذج الموجودة:

| العائلة | الإصدار المستخدم | `features_count` | ملاحظات |
|---|---|---|---|
| `pos_speed` | `20250922-141757` | 58 | يحتوي `head_pos_speed.joblib` و `meta.joblib`. |
| `replay_stale` | `20250922-144601` | 59 | يستخدم `bin_calib.joblib` و `meta.joblib`. |
| `dos` | `20250922-150122` | 59 | يستخدم `bin_calib.joblib` و `meta.joblib`. |
| `sybil` | `20250922-151415` | 59 | يستخدم `bin_calib.joblib` و `meta.joblib`. |

نوع `meta.joblib`:

| العائلة | نوع `meta.joblib` | عرض الإدخال |
|---|---|---|
| `pos_speed` | `sklearn.linear_model.LogisticRegression` | 2 مدخلات: `p_binary`, `p_head`. |
| `replay_stale` | `sklearn.linear_model.LogisticRegression` | 1 مدخل غالباً: `p_binary`. |
| `dos` | `sklearn.linear_model.LogisticRegression` | 1 مدخل غالباً: `p_binary`. |
| `sybil` | `sklearn.linear_model.LogisticRegression` | 1 مدخل غالباً: `p_binary`. |

القراءة في الكود:

| الدالة | السطر | ماذا تفعل |
|---|---|---|
| `latest_archived_family_bundles` | 2083 | تختار أحدث نموذج لكل عائلة، وتثبت `pos_speed` على إصدار مفضل. |
| `LegacyFamilyRuntime` | 2485 | يقرأ نموذج عائلة واحدة. |
| `score_components` | 2584 | يخرج `p_binary`, `p_head`, و `p_final` من `meta.joblib`. |
| `ArchivedFamilyEnsembleRuntime` | 2638 | يجمع نماذج العائلات في Runtime واحد. |

شرحها:

> النماذج المؤرشفة بنيت كحزم لكل عائلة في Colab. عند التشغيل لا أقرأ أرقام الأداء من الملفات، بل أحمّل النماذج نفسها وأطبقها على بيانات السيناريو الحالية.

## 25. كيف تعمل `LegacyFamilyRuntime`

| الجزء | السطر | التفاصيل |
|---|---|---|
| تحميل metadata | 2490 | قراءة `model_meta.json`. |
| تحميل preprocessing | 2494 | قراءة `preproc.joblib`. |
| تحميل binary calibrator | 2495 | قراءة `bin_calib.joblib`. |
| تحميل head إن وجد | 2496 إلى 2499 | مثل `head_pos_speed.joblib`. |
| تجهيز خصائص متوافقة | 2506 | `_build_feature_candidates`. |
| بناء matrix | 2569 | `_build_model_matrix`. |
| حساب المكونات | 2584 | `score_components`. |
| Logistic meta | 2598 إلى 2607 | بناء مدخل `meta.joblib` وإخراج `p_final`. |

في `score_components`:

| الخرج | معناه |
|---|---|
| `p_binary` | احتمال النموذج الثنائي العام داخل العائلة. |
| `p_head` | احتمال الرأس الصريح إن وجد، وإلا يساوي `p_binary`. |
| `p_final` | خرج Logistic Regression الخاص بالعائلة. |
| `meta_width` | عدد مداخل Logistic Regression. |
| `has_explicit_head` | هل يوجد `head_*.joblib`. |

## 26. الدمج في النماذج المؤرشفة

الفئة: `ArchivedFamilyEnsembleRuntime` في `vanet_ids_rsu_core.py:2638`.

خطواتها:

| المرحلة | السطر | العمل |
|---|---|---|
| تحميل حزم العائلات | 2641 إلى 2649 | تحميل `LegacyFamilyRuntime` لكل عائلة. |
| إعداد الثقة | 2651 إلى 2659 | `base_threshold=0.370`, `sensitivity=0.4`. |
| Manifest | 2660 إلى 2680 | يصف الرؤوس والحزم المستخدمة. |
| ضبط الدرجة | 2684 | `_clip_score` يحصر الاحتمال بين 0 و1. |
| حماية من التشبع | 2688 | `_fusion_contribution` يمنع رأساً مشبعاً من جعل كل الصفوف هجوماً. |
| تشغيل العائلات | 2706 | `_score_family_components`. |
| الكشف الكامل | 2712 | `score_dataframe`. |

تفاصيل `score_dataframe`:

| المرحلة | السطر | ماذا يحدث |
|---|---|---|
| 1 | 2713 إلى 2719 | تجهيز البيانات: normalization, features, OBU, Sybil. |
| 2 | 2721 إلى 2723 | تشغيل نماذج العائلات المؤرشفة. |
| 3 | 2725 إلى 2739 | بناء أعمدة `p_general`, `p_pos_speed`, `p_replay_stale`, `p_dos`, `p_sybil`, `p_integrity`. |
| 4 | 2741 إلى 2753 | دمج الاحتمالات في `p_final`. |
| 5 | 2755 وما بعدها | تطبيق الثقة والقرار النهائي. |

صيغة الدمج الحالية:

```text
p_final = max(
  fusion(p_general),
  fusion(p_pos_speed),
  fusion(p_replay_stale),
  fusion(p_dos),
  fusion(p_dos_iforest),
  fusion(p_sybil),
  0.3 * p_integrity
)
```

شرح `_fusion_contribution`:

| الحالة | النتيجة |
|---|---|
| إذا أغلب القيم عالية جداً | يعيد صفراً حتى لا يصنف كل شيء كهجوم. |
| إذا الوسط منخفض لكن توجد قمم 0.99 | يحافظ على القمم العالية فقط. |
| إذا 95% من القيم تحت 0.37 ولا توجد قمم قوية | يعيد صفراً. |
| غير ذلك | يستخدم القيم كما هي. |

شرحها:

> هذا الدمج مختلف عن Logistic Regression الرسمي للحزمة الحديثة. هنا أتعامل مع نماذج عائلات مؤرشفة، لذلك أستخدم دمجاً محافظاً يمنع النموذج المشبع من رفع كل الصفوف إلى هجوم.

## 27. الثقة والقرار النهائي

الفئة: `AdaptiveTrustManager` في `vanet_ids_rsu_core.py:1145`.

| الدالة | السطر | العمل |
|---|---|---|
| `trust` | 1163 | يحسب الثقة `alpha / (alpha + beta)`. |
| `update` | 1167 | يزيد `alpha` للسلوك الجيد أو `beta` للسلوك السيئ. |
| `has_obu_flags` | 1175 | يفحص هل الصف فيه أعلام OBU. |
| `threshold` | 1178 | يحسب العتبة حسب الثقة. |
| `update_after_decision` | 1183 | يحدث الثقة بعد القرار. |

صيغة العتبة:

```text
threshold = clip(base_threshold + sensitivity * (trust - 0.5), floor, ceil)
```

الإعدادات الحالية في النماذج المؤرشفة:

| الإعداد | القيمة |
|---|---|
| `base_threshold` | 0.370 |
| `sensitivity` | 0.4 |
| `floor` | 0.35 |
| `ceil` | 0.85 |
| `w_bad` | 1.0 |
| `w_good` | 0.5 |
| `w_bad_minor` | 0.2 |

شرحها:

> إذا زاد الشك في مركبة تنخفض موثوقيتها، وإذا بقي سلوكها جيداً تتحسن الثقة. القرار النهائي هو مقارنة `p_final` مع العتبة التكيفية الخاصة بالمرسل.

## 28. كشف Offline

الدالة: `run_detect_offline` في `vanet_ids_v2.py:1368`.

| المرحلة | السطر | التفاصيل |
|---|---|---|
| تحميل Runtime | 1378 | `load_runtime(models_dir)`. |
| تحديد نوع المصدر | 1379 | CSV أو raw-dir. |
| إنشاء مجلد الخرج | 1380 إلى 1382 | إذا لم يكن موجوداً. |
| قراءة raw-dir | 1386 إلى 1389 | `load_raw_bsm_directory` ثم `score_dataframe`. |
| قراءة CSV | 1390 إلى 1393 | `pd.read_csv` ثم `score_dataframe`. |
| حساب الأداء | 1397 إلى 1401 | `_detection_scorecard_from_dataframe`. |
| عرض قرارات المركبات | 1402 | `_log_vehicle_decision_table`. |

مسار `load_runtime` في `vanet_ids_rsu_core.py:2250`:

| الترتيب | النوع |
|---|---|
| 1 | `RSUMultiHeadRuntime` للحزمة الحديثة. |
| 2 | `ScenarioLGBMRuntime` للحزمة الصغيرة. |
| 3 | `ArchivedFamilyEnsembleRuntime` للنماذج المؤرشفة متعددة العائلات. |
| 4 | `LegacySimpleRuntime` إذا سمح المستخدم. |
| 5 | `LegacyFamilyRuntime` لحزمة عائلة واحدة. |

شرحها:

> في offline أستخدم نفس خط المعالجة على ملف كامل. هذا يسمح بحساب الأداء لأن الملف يحتوي `label` و `attack_id`.

## 29. كشف Live

الدالة: `run_detect_live` في `vanet_ids_v2.py:1426`.

| المرحلة | السطر | التفاصيل |
|---|---|---|
| تحميل Runtime | 1438 | نفس `load_runtime` المستخدم في offline. |
| قراءة مصدر live | 1407 | `_read_detection_source`. |
| قراءة المفاتيح السابقة | 1415 | `_existing_live_row_keys` يمنع تكرار الصفوف. |
| Polling | 1451 إلى 1467 | يقرأ المصدر كل فترة. |
| الكشف | 1454 | `runtime.score_dataframe(raw_df)`. |
| حفظ الجديد فقط | 1457 إلى 1462 | يضيف فقط row keys الجديدة إلى CSV. |
| عرض الأداء والقرارات | 1469 إلى 1476 | نفس شكل offline عند انتهاء الجولة. |

الفرق بين offline وlive:

| النقطة | Offline | Live |
|---|---|---|
| الإدخال | ملف CSV أو raw-dir ثابت | مصدر يتغير مع الوقت. |
| القراءة | مرة واحدة | polling متكرر. |
| الحفظ | يكتب كل النتائج | يضيف الصفوف الجديدة فقط. |
| الأداء | يظهر إذا توجد labels | يظهر إذا المصدر يحتوي labels. |
| القرار | نفس `score_dataframe` | نفس `score_dataframe`. |

شرحها:

> `detect-live` لا يستخدم منطقاً مختلفاً عن offline. هو فقط يكرر القراءة من مصدر حي ويضيف الصفوف الجديدة. لذلك المعالجة، الرؤوس، `p_final`، والثقة كلها متوافقة مع offline.

## 30. حساب الأداء

| الدالة | السطر | العمل |
|---|---|---|
| `_detection_scorecard_from_dataframe` | `vanet_ids_v2.py:842` | يحسب `TP`, `FP`, `FN`, `TN`, `F1`, `Precision`, `Recall`, `FPR`. |
| `_log_detection_scorecard` | `vanet_ids_v2.py:948` | يعرض جدول `Family Performance`. |
| `_vehicle_decision_rows` | `vanet_ids_v2.py:994` | يجمع النتائج على مستوى المركبة. |
| `_log_vehicle_decision_table` | `vanet_ids_v2.py:1023` | يعرض جدول `Vehicle Decisions`. |
| `run_verify` | `vanet_ids_v2.py:1550` | يقارن labels مع detection CSV. |

مقاييس الأداء:

| المقياس | المعنى |
|---|---|
| `TP` | هجوم حقيقي تم كشفه. |
| `FP` | صف طبيعي تم تصنيفه كهجوم. |
| `FN` | هجوم لم يتم كشفه. |
| `TN` | صف طبيعي تم تصنيفه بشكل صحيح. |
| `Precision` | نسبة الإنذارات الصحيحة من كل الإنذارات. |
| `Recall` | نسبة الهجمات المكتشفة من كل الهجمات الحقيقية. |
| `F1` | توازن بين Precision وRecall. |
| `FPR` | الإنذارات الكاذبة على الطبيعي. |

## 31. تجميع النتائج على مستوى المركبات

الدالة: `_vehicle_decision_rows` في `vanet_ids_v2.py:994`.

طريقة التجميع:

| العمود | كيف يتم حسابه |
|---|---|
| `vehicle` | `sender_pseudo`. |
| `trust_score` | آخر قيمة ثقة للمركبة من `trust_sender_after`. |
| `p_final` | أعلى `p_final` للمركبة. |
| `final_decision` | إذا أي صف للمركبة هجوم، تصبح المركبة `ATTACK`. |

شرحها:

> هذا هو التجميع الثاني في النظام. بعد قرار كل رسالة، أجمع الرسائل على مستوى المركبة حتى أعرض حالة كل مركبة أمام اللجنة.

## 32. الأعمدة النهائية في CSV الكشف

| العمود | المعنى |
|---|---|
| `row_id` | رقم الصف. |
| `row_key` | مفتاح ثابت للمطابقة والتحقق. |
| `receiver_pseudo` | المستقبل. |
| `sender_pseudo` | المركبة المرسلة. |
| `t_curr` | الزمن الحالي. |
| `dt` | فرق الزمن عن الرسالة السابقة لنفس المركبة. |
| `label` | الملصق الحقيقي إن وجد. |
| `attack_id` | رقم الهجوم إن وجد. |
| `flag_*` | أعلام OBU. |
| `obu_flag_count` | عدد أعلام OBU. |
| `obu_risk` | خطر OBU النسبي. |
| `obu_anom` | قرار OBU أولي. |
| `p_general` | احتمال الرأس العام. |
| `p_pos_speed` | احتمال الموقع/السرعة. |
| `p_replay_stale` | احتمال replay/stale. |
| `p_dos` | احتمال DoS. |
| `p_dos_iforest` | درجة شذوذ DoS. |
| `p_sybil` | احتمال Sybil. |
| `p_integrity` | احتمال أو خطر التكامل. |
| `p_final` | الاحتمال النهائي بعد الدمج. |
| `adaptive_threshold` | العتبة التكيفية. |
| `trust_sender_before` | الثقة قبل تحديث الصف. |
| `trust_sender_after` | الثقة بعد تحديث الصف. |
| `final_decision` | القرار النهائي للصف. |
| `runtime_mode` | نوع Runtime المستخدم. |

## 33. التدريب البديل للسيناريو الصغير

إذا كانت بيانات السيناريو صغيرة ولا تكفي لتقسيم sender-disjoint الكامل، يستخدم `run_train_rsu` fallback:

| العنصر | المكان |
|---|---|
| fallback في CLI | `vanet_ids_v2.py:1496` إلى `1523` |
| التدريب الصغير | `train_scenario_lgbm_release` في `vanet_ids_rsu_core.py:2861` |
| Runtime الخاص به | `ScenarioLGBMRuntime` في `vanet_ids_rsu_core.py:2955` |

شرحها:

> هذا fallback للتجارب الصغيرة فقط. الحزمة الأقوى للنقاش هي متعددة الرؤوس أو النماذج المؤرشفة متعددة العائلات.

## 34. نقاط جاهزة للشرح أمام اللجنة

| السؤال | الجواب المختصر |
|---|---|
| لماذا تستخدم أكثر من رأس؟ | لأن كل عائلة هجوم لها نمط مختلف؛ DoS يعتمد على معدل الرسائل، Sybil على الهوية، وpos_speed على الحركة. |
| لماذا Logistic Regression؟ | لأنه meta-classifier يأخذ احتمالات الرؤوس ويتعلم وزنها لإنتاج `p_final`. |
| لماذا OOF؟ | حتى لا يتعلم meta-classifier من توقعات متحيزة على نفس بيانات تدريب الرأس. |
| لماذا sender-disjoint؟ | لمنع تسرب هوية المركبة بين التدريب والاختبار. |
| لماذا توجد OBU flags؟ | لأنها تكشف المخالفات الفيزيائية والبروتوكولية قبل الاعتماد الكامل على النموذج. |
| لماذا trust score؟ | لأن سلوك المركبة عبر الزمن أهم من رسالة واحدة. |
| لماذا live وoffline متوافقان؟ | لأن الاثنين يستخدمان `runtime.score_dataframe` نفسه؛ الاختلاف فقط في طريقة قراءة البيانات. |
| لماذا `FPR` مهم؟ | لأن الإنذار الكاذب في VANET قد يؤدي إلى عزل مركبة سليمة. |

## 35. نص شرح كامل مختصر

> بدأت بالبنية التحتية: F2MD وSUMO لتوليد رسائل BSM، ثم استخرجت الرسائل إلى CSV. بعد ذلك أضفت `attack_id` حتى أعرف نوع الهجوم في التقييم. قبل التدريب أو الكشف يمر كل صف بمرحلة تطبيع، ثم هندسة خصائص حركية وزمنية، ثم خصائص Sybil، ثم أعلام OBU. في التدريب أستخدم تقسيم sender-disjoint حتى لا تظهر نفس المركبة في التدريب والاختبار. الرؤوس المتخصصة تتعلم أنماط العائلات المختلفة، وبعدها يتم تدريب Logistic Regression كـ meta-classifier على احتمالات الرؤوس لإنتاج `p_final`. في الكشف offline أو live يتم تحميل نفس النماذج وتشغيل نفس `score_dataframe`. أخيراً يتم تحديث ثقة كل مركبة وتحديد القرار النهائي، ثم أعرض أداء العائلة وجدول قرارات المركبات.

## 36. أوامر عملية مهمة

تدريب من CLI:

```bash
python vanet_ids_v2.py train-rsu \
  --input /path/to/labeled.csv \
  --models_dir /path/to/release_v3 \
  --train_family all
```

كشف offline:

```bash
python vanet_ids_v2.py detect-offline \
  --models_dir "/home/instantf2md/Desktop/VANET_project/results and models/models" \
  --input /home/instantf2md/F2MD/f2md-results/features_intermessage_v2.csv \
  --output /home/instantf2md/F2MD/f2md-results/detect_offline.csv \
  --source_kind csv
```

كشف live مرة واحدة:

```bash
python vanet_ids_v2.py detect-live \
  --models_dir "/home/instantf2md/Desktop/VANET_project/results and models/models" \
  --input /home/instantf2md/F2MD/f2md-results/features_intermessage_v2.csv \
  --output /home/instantf2md/F2MD/f2md-results/detect_live.csv \
  --source_kind csv \
  --once
```

تحقق:

```bash
python vanet_ids_v2.py verify \
  --labels /path/to/labeled.csv \
  --detect_csv /path/to/detect.csv \
  --outdir /path/to/verify_output
```

