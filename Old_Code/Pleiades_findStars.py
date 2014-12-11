files = ['./4259_Pleaides/aspcapStar-v304-2m03403073+2429143.fits', './4259_Pleaides/aspcapStar-v304-2m03405126+2335544.fits', './4259_Pleaides/aspcapStar-v304-2m03415366+2327287.fits', './4259_Pleaides/aspcapStar-v304-2m03415868+2342263.fits', './4259_Pleaides/aspcapStar-v304-2m03420383+2442454.fits', './4259_Pleaides/aspcapStar-v304-2m03422154+2439527.fits', './4259_Pleaides/aspcapStar-v304-2m03422760+2502492.fits', './4259_Pleaides/aspcapStar-v304-2m03432662+2459395.fits', './4259_Pleaides/aspcapStar-v304-2m03433660+2327141.fits', './4259_Pleaides/aspcapStar-v304-2m03433692+2423382.fits', './4259_Pleaides/aspcapStar-v304-2m03434860+2332218.fits', './4259_Pleaides/aspcapStar-v304-2m03435214+2450297.fits', './4259_Pleaides/aspcapStar-v304-2m03440509+2529017.fits', './4259_Pleaides/aspcapStar-v304-2m03443742+2508161.fits', './4259_Pleaides/aspcapStar-v304-2m03445017+2454401.fits', './4259_Pleaides/aspcapStar-v304-2m03445896+2323202.fits', './4259_Pleaides/aspcapStar-v304-2m03451199+2435102.fits', './4259_Pleaides/aspcapStar-v304-2m03452219+2328182.fits', './4259_Pleaides/aspcapStar-v304-2m03453903+2513279.fits', './4259_Pleaides/aspcapStar-v304-2m03454245+2503255.fits', './4259_Pleaides/aspcapStar-v304-2m03460381+2527108.fits', './4259_Pleaides/aspcapStar-v304-2m03460525+2258540.fits', './4259_Pleaides/aspcapStar-v304-2m03460649+2434027.fits', './4259_Pleaides/aspcapStar-v304-2m03460777+2452005.fits', './4259_Pleaides/aspcapStar-v304-2m03461175+2437203.fits', './4259_Pleaides/aspcapStar-v304-2m03462047+2447077.fits', './4259_Pleaides/aspcapStar-v304-2m03462863+2445324.fits', './4259_Pleaides/aspcapStar-v304-2m03463533+2324422.fits', './4259_Pleaides/aspcapStar-v304-2m03463727+2420367.fits', './4259_Pleaides/aspcapStar-v304-2m03463777+2444517.fits', './4259_Pleaides/aspcapStar-v304-2m03463888+2431132.fits', './4259_Pleaides/aspcapStar-v304-2m03464027+2455517.fits', './4259_Pleaides/aspcapStar-v304-2m03464831+2418060.fits', './4259_Pleaides/aspcapStar-v304-2m03471481+2522186.fits', './4259_Pleaides/aspcapStar-v304-2m03471806+2423268.fits', './4259_Pleaides/aspcapStar-v304-2m03472083+2505124.fits', './4259_Pleaides/aspcapStar-v304-2m03473368+2441032.fits', './4259_Pleaides/aspcapStar-v304-2m03473521+2532383.fits', './4259_Pleaides/aspcapStar-v304-2m03473801+2328050.fits', './4259_Pleaides/aspcapStar-v304-2m03475973+2443528.fits', './4259_Pleaides/aspcapStar-v304-2m03481018+2300041.fits', './4259_Pleaides/aspcapStar-v304-2m03481099+2330253.fits', './4259_Pleaides/aspcapStar-v304-2m03481729+2430159.fits', './4259_Pleaides/aspcapStar-v304-2m03482277+2358212.fits']

values = [8.11894956, 8.11207383, 8.13607165, 9.95616439, 8.09061697, 9.97656862, 8.1227628, 8.18651006, 8.11408725, 8.05059853, 8.15764743, 8.1059421, 8.10766008, 8.13074422, 8.12143394, 8.11297022, 8.12831998, 8.1340771, 10.14725175, 8.11899054, 8.12952193, 8.15221549, 8.1356128, 8.23349497, 8.12809746, 8.13079917, 8.1387121, 9.99598242, 8.14077088, 8.20355362, 8.12313408, 8.16263621, 8.07449536, 8.07038524, 8.15365346, 10.23576383, 8.13941618, 9.91912385, 8.13421706, 9.81563788, 8.12090039, 8.12390428, 8.13512813, 10.21895611]

print len(files)
print len(values)

# Find the filenames corresponding to the misbehaving stars

stars = []

for i in range(0, len(values)):
    if values[i] > 9:
        stars.append(files[i])

print stars

