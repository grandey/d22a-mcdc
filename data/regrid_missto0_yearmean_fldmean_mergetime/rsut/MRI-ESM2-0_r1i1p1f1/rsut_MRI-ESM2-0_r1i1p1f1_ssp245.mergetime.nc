CDF   �   
      time       bnds      lon       lat          .   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       EMRI-ESM2.0 (2017): 
aerosol: MASINGAR mk2r4 (TL95; 192 x 96 longitude/latitude; 80 levels; top level 0.01 hPa)
atmos: MRI-AGCM3.5 (TL159; 320 x 160 longitude/latitude; 80 levels; top level 0.01 hPa)
atmosChem: MRI-CCM2.1 (T42; 128 x 64 longitude/latitude; 80 levels; top level 0.01 hPa)
land: HAL 1.0
landIce: none
ocean: MRI.COM4.4 (tripolar primarily 0.5 deg latitude/1 deg longitude with meridional refinement down to 0.3 deg within 10 degrees north and south of the equator; 360 x 364 longitude/latitude; 61 levels; top grid cell 0-2 m)
ocnBgchem: MRI.COM4.4
seaIce: MRI.COM4.4      institution       CMeteorological Research Institute, Tsukuba, Ibaraki 305-0052, Japan    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2019-02-20T02:45:03Z   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Lhttps://furtherinfo.es-doc.org/CMIP6.MRI.MRI-ESM2-0.historical.none.r1i1p1f1   grid      7native atmosphere TL159 gaussian grid (160x320 latxlon)    
grid_label        gn     history      	�Wed Aug 10 15:19:25 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/MRI-ESM2-0_r1i1p1f1/rsut_MRI-ESM2-0_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.ScenarioMIP.MRI.MRI-ESM2-0.ssp245.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_ssp245_r1i1p1f1_gn_201501-210012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/MRI-ESM2-0_r1i1p1f1/rsut_MRI-ESM2-0_r1i1p1f1_ssp245.mergetime.nc
Wed Aug 10 15:19:24 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/MRI-ESM2-0_r1i1p1f1/rsut_MRI-ESM2-0_r1i1p1f1_historical.mergetime.nc
Fri Apr 08 10:34:21 2022: cdo -O -s -selname,rsut -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 10:34:16 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rsut -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc
2019-02-20T02:45:03Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.;
Output from run-Dr060_historical_101 (sfc_avr_mon.ctl)     initialization_index            institution_id        MRI    mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      
MRI-ESM2-0     parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      	source_id         
MRI-ESM2-0     source_type       AOGCM AER CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(14 December 2018) MD5:b2d32d1a0d9b196411429c8895329d8f      title         $MRI-ESM2-0 output prepared for CMIP6   variable_id       rsut   variant_label         r1i1p1f1   license      CMIP6 model data produced by MRI is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.4.0      tracking_id       1hdl:21.14100/405a2b90-dfa1-4482-97dc-1f18ac879500      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T                   	time_bnds                                 (   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X                  lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y                  rsut                   
   standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       at the top of the atmosphere   original_name         USWT   cell_measures         area: areacella    history       r2019-02-20T02:45:03Z altered by CMOR: replaced missing value flag (-9.99e+33) with standard missing value (1e+20).              8                Aq���   Aq��P   Aq�P   B��hAq�6�   Aq�P   Aq��P   B;Aq���   Aq��P   Aq��P   B�=:Aq��   Aq��P   Aq�dP   B�N�Aq���   Aq�dP   Aq��P   B�KaAq���   Aq��P   Aq�FP   B��QAq�k�   Aq�FP   Aq��P   B�O�Aq���   Aq��P   Aq�(P   B�QAq�M�   Aq�(P   Aq��P   B�>PAq���   Aq��P   Aq�
P   B� �Aq�/�   Aq�
P   Aq�{P   B���Aq���   Aq�{P   Aq��P   B�OAq��   Aq��P   Aq�]P   B��AqĂ�   Aq�]P   Aq��P   B�Z=Aq���   Aq��P   Aq�?P   B�0�Aq�d�   Aq�?P   Aq˰P   B�m�Aq���   Aq˰P   Aq�!P   B�zkAq�F�   Aq�!P   AqВP   B���Aqз�   AqВP   Aq�P   BÆ�Aq�(�   Aq�P   Aq�tP   B���Aqՙ�   Aq�tP   Aq��P   BÕ%Aq�
�   Aq��P   Aq�VP   B��oAq�{�   Aq�VP   Aq��P   B�t�Aq���   Aq��P   Aq�8P   B���Aq�]�   Aq�8P   Aq�P   B���Aq���   Aq�P   Aq�P   B�OFAq�?�   Aq�P   Aq�P   Bç�Aq��   Aq�P   Aq��P   B��Aq�!�   Aq��P   Aq�mP   B���Aq��   Aq�mP   Aq��P   B�o�Aq��   Aq��P   Aq�OP   B��Aq�t�   Aq�OP   Aq��P   B�vsAq���   Aq��P   Aq�1P   B�ZAq�V�   Aq�1P   Aq��P   B�q`Aq���   Aq��P   Aq�P   BȖ�Aq�8�   Aq�P   Aq��P   B��Aq���   Aq��P   Aq��P   B�&�Aq��   Aq��P   ArfP   B�O�Ar��   ArfP   Ar�P   B�k�Ar��   Ar�P   ArHP   B�E/Arm�   ArHP   Ar�P   B�ѲAr��   Ar�P   Ar*P   B�J�ArO�   Ar*P   Ar�P   BÀ[Ar��   Ar�P   ArP   Bê|Ar1�   ArP   Ar}P   B�O�Ar��   Ar}P   Ar�P   B��Ar�   Ar�P   Ar_P   B�QAr��   Ar_P   Ar�P   BÔ�Ar��   Ar�P   ArAP   B�22Arf�   ArAP   Ar�P   BĮ�Ar��   Ar�P   Ar!#P   BÇ�Ar!H�   Ar!#P   Ar#�P   B�Ar#��   Ar#�P   Ar&P   B�SAr&*�   Ar&P   Ar(vP   B�]�Ar(��   Ar(vP   Ar*�P   B�p�Ar+�   Ar*�P   Ar-XP   B�DBAr-}�   Ar-XP   Ar/�P   B�\�Ar/��   Ar/�P   Ar2:P   B�=Ar2_�   Ar2:P   Ar4�P   Bï�Ar4��   Ar4�P   Ar7P   B��oAr7A�   Ar7P   Ar9�P   B�S�Ar9��   Ar9�P   Ar;�P   BĆ�Ar<#�   Ar;�P   Ar>oP   B�1Ar>��   Ar>oP   Ar@�P   B�\JArA�   Ar@�P   ArCQP   B��iArCv�   ArCQP   ArE�P   B�R.ArE��   ArE�P   ArH3P   B�pArHX�   ArH3P   ArJ�P   Bľ�ArJ��   ArJ�P   ArMP   BĎArM:�   ArMP   ArO�P   B�ϷArO��   ArO�P   ArQ�P   B��ArR�   ArQ�P   ArThP   BĺXArT��   ArThP   ArV�P   B��ArV��   ArV�P   ArYJP   B�'�ArYo�   ArYJP   Ar[�P   B��Ar[��   Ar[�P   Ar^,P   B�UsAr^Q�   Ar^,P   Ar`�P   B��uAr`��   Ar`�P   ArcP   Bď�Arc3�   ArcP   AreP   BĒ�Are��   AreP   Arg�P   Bě�Arh�   Arg�P   ArjaP   B��#Arj��   ArjaP   Arl�P   B��Arl��   Arl�P   AroCP   B�0�Aroh�   AroCP   Arq�P   BĨ�Arq��   Arq�P   Art%P   B��IArtJ�   Art%P   Arv�P   B�Arv��   Arv�P   AryP   B��Ary,�   AryP   Ar{xP   B�M#Ar{��   Ar{xP   Ar}�P   B�YCAr~�   Ar}�P   Ar�ZP   B�ԊAr��   Ar�ZP   Ar��P   B�r�Ar���   Ar��P   Ar�<P   B��Ar�a�   Ar�<P   Ar��P   B�Q�Ar���   Ar��P   Ar�P   B�#CAr�C�   Ar�P   Ar��P   BıPAr���   Ar��P   Ar� P   B�uAr�%�   Ar� P   Ar�qP   Bö�Ar���   Ar�qP   Ar��P   B�"�Ar��   Ar��P   Ar�SP   B��<Ar�x�   Ar�SP   Ar��P   B��$Ar���   Ar��P   Ar�5P   BĊ0Ar�Z�   Ar�5P   Ar��P   B��AAr���   Ar��P   Ar�P   B�R�Ar�<�   Ar�P   Ar��P   B�FpAr���   Ar��P   Ar��P   B�1Ar��   Ar��P   Ar�jP   Bſ7Ar���   Ar�jP   Ar��P   Bƀ�Ar� �   Ar��P   Ar�LP   B�	�Ar�q�   Ar�LP   Ar��P   B��FAr���   Ar��P   Ar�.P   B�C�Ar�S�   Ar�.P   Ar��P   B�+Ar���   Ar��P   Ar�P   B�D�Ar�5�   Ar�P   Ar��P   B�o�Ar���   Ar��P   Ar��P   Bȵ�Ar��   Ar��P   Ar�cP   B��IAr���   Ar�cP   Ar��P   BǫvAr���   Ar��P   Ar�EP   B��,Ar�j�   Ar�EP   ArĶP   BǣAr���   ArĶP   Ar�'P   B�\Ar�L�   Ar�'P   ArɘP   BǹArɽ�   ArɘP   Ar�	P   B��Ar�.�   Ar�	P   Ar�zP   BǕ�ArΟ�   Ar�zP   Ar��P   B��gAr��   Ar��P   Ar�\P   B��
ArӁ�   Ar�\P   Ar��P   B�M�Ar���   Ar��P   Ar�>P   B�&�Ar�c�   Ar�>P   ArگP   B�24Ar���   ArگP   Ar� P   Bǽ�Ar�E�   Ar� P   ArߑP   B�LJAr߶�   ArߑP   Ar�P   BȪ�Ar�'�   Ar�P   Ar�sP   B�d"Ar��   Ar�sP   Ar��P   B�WhAr�	�   Ar��P   Ar�UP   B��Ar�z�   Ar�UP   Ar��P   Bʾ�Ar���   Ar��P   Ar�7P   BȋAr�\�   Ar�7P   Ar�P   B�VAr���   Ar�P   Ar�P   B�-�Ar�>�   Ar�P   Ar��P   B�p�Ar���   Ar��P   Ar��P   BȻAr� �   Ar��P   Ar�lP   B�yyAr���   Ar�lP   Ar��P   B�I�Ar��   Ar��P   Ar�NP   B��Ar�s�   Ar�NP   As�P   B�*�As��   As�P   As0P   B��AsU�   As0P   As�P   BǺAs��   As�P   As	P   B���As	7�   As	P   As�P   B���As��   As�P   As�P   B� As�   As�P   AseP   BǼvAs��   AseP   As�P   B��-As��   As�P   AsGP   B� `Asl�   AsGP   As�P   B��~As��   As�P   As)P   BǀAsN�   As)P   As�P   B�DAs��   As�P   AsP   B�eYAs0�   AsP   As!|P   Bǽ�As!��   As!|P   As#�P   BǭAs$�   As#�P   As&^P   B�*As&��   As&^P   As(�P   B�cmAs(��   As(�P   As+@P   B�f�As+e�   As+@P   As-�P   B�As-��   As-�P   As0"P   B�x4As0G�   As0"P   As2�P   BƼsAs2��   As2�P   As5P   B��"As5)�   As5P   As7uP   Bƺ�As7��   As7uP   As9�P   B�X�As:�   As9�P   As<WP   B�oQAs<|�   As<WP   As>�P   B�D�As>��   As>�P   AsA9P   B�M�AsA^�   AsA9P   AsC�P   B��AsC��   AsC�P   AsFP   B��qAsF@�   AsFP   AsH�P   BôRAsH��   AsH�P   AsJ�P   B�RAsK"�   AsJ�P   AsMnP   BŴ�AsM��   AsMnP   AsO�P   B�3AsP�   AsO�P   AsRPP   B��AsRu�   AsRPP   AsT�P   B���AsT��   AsT�P   AsW2P   Bŧ}AsWW�   AsW2P   AsY�P   B�(�AsY��   AsY�P   As\P   Bũ�As\9�   As\P   As^�P   B�y�As^��   As^�P   As`�P   BĂHAsa�   As`�P   AscgP   B�MaAsc��   AscgP   Ase�P   B�B�Ase��   Ase�P   AshIP   B��9Ashn�   AshIP   Asj�P   BĿFAsj��   Asj�P   Asm+P   B���AsmP�   Asm+P   Aso�P   B���Aso��   Aso�P   AsrP   B�6;Asr2�   AsrP   Ast~P   B��CAst��   Ast~P   Asv�P   B��Asw�   Asv�P   Asy`P   B�VAsy��   Asy`P   As{�P   BéoAs{��   As{�P   As~BP   Bè�As~g�   As~BP   As��P   B��As���   As��P   As�$P   B¥�As�I�   As�$P   As��P   B<As���   As��P   As�P   B�{�As�+�   As�P   As�wP   B�kFAs���   As�wP   As��P   B�As��   As��P   As�YP   B��AAs�~�   As�YP   As��P   B�5nAs���   As��P   As�;P   B�A�As�`�   As�;P   As��P   B�&�As���   As��P   As�P   B��As�B�   As�P   As��P   B�K�As���   As��P   As��P   B�L�As�$�   As��P   As�pP   B�As���   As�pP   As��P   B��aAs��   As��P   As�RP   B��As�w�   As�RP   As��P   B�S�As���   As��P   As�4P   B�*As�Y�   As�4P   As��P   B�LoAs���   As��P   As�P   B�OAs�;�   As�P   As��P   B�Q#As���   As��P   As��P   B�P�As��   As��P   As�iP   B��+As���   As�iP   As��P   B�K�As���   As��P   As�KP   B�F�As�p�   As�KP   As��P   B��As���   As��P   As�-P   B�mAs�R�   As�-P   AsP   B��As���   AsP   As�P   B��DAs�4�   As�P   AsǀP   B��|Asǥ�   AsǀP   As��P   B��As��   As��P   As�bP   B�=Aṡ�   As�bP   As��P   B�CAs���   As��P   As�DP   B�}As�i�   As�DP   AsӵP   B��As���   AsӵP   As�&P   B�PAs�K�   As�&P   AsؗP   B�ғAsؼ�   AsؗP   As�P   B��As�-�   As�P   As�yP   B���Asݞ�   As�yP   As��P   B�;�As��   As��P   As�[P   B� ;As��   As�[P   As��P   B�6�As���   As��P   As�=P   B���As�b�   As�=P   As�P   B�8�As���   As�P   As�P   B�"As�D�   As�P   As�P   B�@EAs��   As�P   As�P   B�U�As�&�   As�P   As�rP   B�h�As��   As�rP   As��P   B��WAs��   As��P   As�TP   B��%As�y�   As�TP   As��P   B�t�As���   As��P   As�6P   B�5zAs�[�   As�6P   As��P   B���As���   As��P   AtP   B�l�At=�   AtP   At�P   B�IDAt��   At�P   At�P   B�8)At�   At�P   At	kP   B���