CDF   �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       bACCESS-CM2 (2019): 
aerosol: UKCA-GLOMAP-mode
atmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
atmosChem: none
land: CABLE2.5
landIce: none
ocean: ACCESS-OM2 (GFDL-MOM5, tripolar primarily 1deg; 360 x 300 longitude/latitude; 50 levels; top grid cell 0-10 m)
ocnBgchem: none
seaIce: CICE5.1.2 (same grid as ocean)     institution       �CSIRO (Commonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia), ARCCSS (Australian Research Council Centre of Excellence for Climate System Science)    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2019-11-08T21:42:28Z   data_specs_version        01.00.30   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Uhttps://furtherinfo.es-doc.org/CMIP6.CSIRO-ARCCSS.ACCESS-CM2.historical.none.r1i1p1f1      grid      ,native atmosphere N96 grid (144x192 latxlon)   
grid_label        gn     history      	�Wed Aug 10 15:20:45 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/hfds/ACCESS-CM2_r1i1p1f1/hfds_ACCESS-CM2_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/ACCESS-CM2_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO-ARCCSS.ACCESS-CM2.ssp370.r1i1p1f1.Omon.hfds.gn.v20191108/hfds_Omon_ACCESS-CM2_ssp370_r1i1p1f1_gn_201501-210012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/hfds/ACCESS-CM2_r1i1p1f1/hfds_ACCESS-CM2_r1i1p1f1_ssp370.mergetime.nc
Wed Aug 10 15:20:43 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Omon.hfds.gn.v20191108/hfds_Omon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/hfds/ACCESS-CM2_r1i1p1f1/hfds_ACCESS-CM2_r1i1p1f1_historical.mergetime.nc
Fri Apr 08 02:48:58 2022: cdo -O -s -selname,hfds -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/hfds/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Omon.hfds.gn.v20191108/hfds_Omon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Omon.hfds.gn.v20191108/hfds_Omon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 02:48:52 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,hfds -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/hfds/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Omon.hfds.gn.v20191108/hfds_Omon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/hfds/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Omon.hfds.gn.v20191108/hfds_Omon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/hfds/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Omon.hfds.gn.v20191108/hfds_Omon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc
2019-11-08T21:42:28Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.      initialization_index            institution_id        CSIRO-ARCCSS   mip_era       CMIP6      nominal_resolution        250 km     notes         �Exp: CM2-historical; Local ID: bj594; Variable: hfds (['sfc_hflux_from_runoff', 'sfc_hflux_coupler', 'sfc_hflux_from_water_evap', 'sfc_hflux_from_water_prec', 'frazil_2d'])   parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      
ACCESS-CM2     parent_time_units         days since 0950-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         ocean      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         
ACCESS-CM2     source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         $ACCESS-CM2 output prepared for CMIP6   variable_id       hfds   variant_label         r1i1p1f1   version       	v20191108      cmor_version      3.4.0      tracking_id       1hdl:21.14100/6f1726b3-31a7-4609-af04-cc80b98c5aab      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               t   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               |   hfds                   	   standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any 'flux adjustment') .    cell_measures         area: areacello    history       o2019-11-08T21:42:24Z altered by CMOR: replaced missing value flag (-1e+20) with standard missing value (1e+20).             �                Aq���   Aq��P   Aq�P   >���Aq�6�   Aq�P   Aq��P   >9�Aq���   Aq��P   Aq��P   >DAq��   Aq��P   Aq�dP   �{^=Aq���   Aq�dP   Aq��P   �̵Aq���   Aq��P   Aq�FP   �
�`Aq�k�   Aq�FP   Aq��P   >��Aq���   Aq��P   Aq�(P   =E[Aq�M�   Aq�(P   Aq��P   >t�Aq���   Aq��P   Aq�
P   �E�Aq�/�   Aq�
P   Aq�{P   �Pp�Aq���   Aq�{P   Aq��P   <�aAq��   Aq��P   Aq�]P   ��B�AqĂ�   Aq�]P   Aq��P   =�i�Aq���   Aq��P   Aq�?P   ��1SAq�d�   Aq�?P   Aq˰P   =ʁ5Aq���   Aq˰P   Aq�!P   >Y�@Aq�F�   Aq�!P   AqВP   ��cAqз�   AqВP   Aq�P   ?�DAq�(�   Aq�P   Aq�tP   >��Aqՙ�   Aq�tP   Aq��P   > N�Aq�
�   Aq��P   Aq�VP   >���Aq�{�   Aq�VP   Aq��P   ��R�Aq���   Aq��P   Aq�8P   �L��Aq�]�   Aq�8P   Aq�P   �_�Aq���   Aq�P   Aq�P   >*V�Aq�?�   Aq�P   Aq�P   >�J�Aq��   Aq�P   Aq��P   >��Aq�!�   Aq��P   Aq�mP   >
��Aq��   Aq�mP   Aq��P   >QAq��   Aq��P   Aq�OP   �(E�Aq�t�   Aq�OP   Aq��P   ?L�Aq���   Aq��P   Aq�1P   >��~Aq�V�   Aq�1P   Aq��P   ��G�Aq���   Aq��P   Aq�P   ���@Aq�8�   Aq�P   Aq��P   ����Aq���   Aq��P   Aq��P   >3m}Aq��   Aq��P   ArfP   >�
�Ar��   ArfP   Ar�P   ���!Ar��   Ar�P   ArHP   >�<SArm�   ArHP   Ar�P   >��Ar��   Ar�P   Ar*P   �!o�ArO�   Ar*P   Ar�P   >L?�Ar��   Ar�P   ArP   >ʄAr1�   ArP   Ar}P   �ޥ�Ar��   Ar}P   Ar�P   >��Ar�   Ar�P   Ar_P   >�c�Ar��   Ar_P   Ar�P   �8��Ar��   Ar�P   ArAP   ����Arf�   ArAP   Ar�P   >�!Ar��   Ar�P   Ar!#P   >o/!Ar!H�   Ar!#P   Ar#�P   <k�Ar#��   Ar#�P   Ar&P   =o�Ar&*�   Ar&P   Ar(vP   �?b�Ar(��   Ar(vP   Ar*�P   ��C�Ar+�   Ar*�P   Ar-XP   8<��Ar-}�   Ar-XP   Ar/�P   =D�?Ar/��   Ar/�P   Ar2:P   >�VAr2_�   Ar2:P   Ar4�P   �бaAr4��   Ar4�P   Ar7P   =�ИAr7A�   Ar7P   Ar9�P   ���Ar9��   Ar9�P   Ar;�P   �	��Ar<#�   Ar;�P   Ar>oP   �˂Ar>��   Ar>oP   Ar@�P   ��F�ArA�   Ar@�P   ArCQP   ���ArCv�   ArCQP   ArE�P   =�H�ArE��   ArE�P   ArH3P   �ܴ�ArHX�   ArH3P   ArJ�P   ? �<ArJ��   ArJ�P   ArMP   >�:ArM:�   ArMP   ArO�P   ��ArO��   ArO�P   ArQ�P   >g9�ArR�   ArQ�P   ArThP   >p3�ArT��   ArThP   ArV�P   >��lArV��   ArV�P   ArYJP   =�e�ArYo�   ArYJP   Ar[�P   >TSAr[��   Ar[�P   Ar^,P   >юAr^Q�   Ar^,P   Ar`�P   >�a�Ar`��   Ar`�P   ArcP   >j�	Arc3�   ArcP   AreP   ?-LAre��   AreP   Arg�P   ���Arh�   Arg�P   ArjaP   >�CArj��   ArjaP   Arl�P   >��[Arl��   Arl�P   AroCP   >��Aroh�   AroCP   Arq�P   >�u�Arq��   Arq�P   Art%P   <��3ArtJ�   Art%P   Arv�P   >O�+Arv��   Arv�P   AryP   ��ǐAry,�   AryP   Ar{xP   >�G�Ar{��   Ar{xP   Ar}�P   ?Ar~�   Ar}�P   Ar�ZP   >��uAr��   Ar�ZP   Ar��P   >��Ar���   Ar��P   Ar�<P   >�RiAr�a�   Ar�<P   Ar��P   ��@Ar���   Ar��P   Ar�P   =���Ar�C�   Ar�P   Ar��P   ���Ar���   Ar��P   Ar� P   >}-,Ar�%�   Ar� P   Ar�qP   �(�Ar���   Ar�qP   Ar��P   ?ZAr��   Ar��P   Ar�SP   <��lAr�x�   Ar�SP   Ar��P   ��Ar���   Ar��P   Ar�5P   >��Ar�Z�   Ar�5P   Ar��P   �
d�Ar���   Ar��P   Ar�P   >C�Ar�<�   Ar�P   Ar��P   >�[Ar���   Ar��P   Ar��P   ���zAr��   Ar��P   Ar�jP   �[�&Ar���   Ar�jP   Ar��P   >�(Ar� �   Ar��P   Ar�LP   >5Ar�q�   Ar�LP   Ar��P   =�o�Ar���   Ar��P   Ar�.P   =#��Ar�S�   Ar�.P   Ar��P   ���Ar���   Ar��P   Ar�P   �vO�Ar�5�   Ar�P   Ar��P   �ˌ�Ar���   Ar��P   Ar��P   ��e�Ar��   Ar��P   Ar�cP   ��N`Ar���   Ar�cP   Ar��P   ��CNAr���   Ar��P   Ar�EP   ?#�Ar�j�   Ar�EP   ArĶP   =���Ar���   ArĶP   Ar�'P   ��ՎAr�L�   Ar�'P   ArɘP   �0��Arɽ�   ArɘP   Ar�	P   �q��Ar�.�   Ar�	P   Ar�zP   �x)ArΟ�   Ar�zP   Ar��P   >�z'Ar��   Ar��P   Ar�\P   ��ArӁ�   Ar�\P   Ar��P   =�˽Ar���   Ar��P   Ar�>P   ����Ar�c�   Ar�>P   ArگP   =1��Ar���   ArگP   Ar� P   >k!�Ar�E�   Ar� P   ArߑP   >�Ar߶�   ArߑP   Ar�P   <���Ar�'�   Ar�P   Ar�sP   >Y�Ar��   Ar�sP   Ar��P   >y� Ar�	�   Ar��P   Ar�UP   >(��Ar�z�   Ar�UP   Ar��P   =�=Ar���   Ar��P   Ar�7P   �I)Ar�\�   Ar�7P   Ar�P   =F�;Ar���   Ar�P   Ar�P   �s�Ar�>�   Ar�P   Ar��P   ?!S;Ar���   Ar��P   Ar��P   ?�Ar� �   Ar��P   Ar�lP   ?`�DAr���   Ar�lP   Ar��P   >�Ar��   Ar��P   Ar�NP   �WdBAr�s�   Ar�NP   As�P   �<��As��   As�P   As0P   �O�BAsU�   As0P   As�P   >���As��   As�P   As	P   ?44�As	7�   As	P   As�P   ?.E�As��   As�P   As�P   >�mAs�   As�P   AseP   ?n��As��   AseP   As�P   ?~ZAs��   As�P   AsGP   ?%T?Asl�   AsGP   As�P   ?=SrAs��   As�P   As)P   ?C��AsN�   As)P   As�P   ?S�As��   As�P   AsP   ?�As0�   AsP   As!|P   >��RAs!��   As!|P   As#�P   >UDAs$�   As#�P   As&^P   ?h��As&��   As&^P   As(�P   ?[�\As(��   As(�P   As+@P   ?K��As+e�   As+@P   As-�P   ?FtnAs-��   As-�P   As0"P   >��vAs0G�   As0"P   As2�P   ?#WAs2��   As2�P   As5P   ?��As5)�   As5P   As7uP   ?�f�As7��   As7uP   As9�P   ?��WAs:�   As9�P   As<WP   ?B��As<|�   As<WP   As>�P   >|�sAs>��   As>�P   AsA9P   ?l�AsA^�   AsA9P   AsC�P   ?U=�AsC��   AsC�P   AsFP   ?���AsF@�   AsFP   AsH�P   ?t��AsH��   AsH�P   AsJ�P   ?�M�AsK"�   AsJ�P   AsMnP   ?��AsM��   AsMnP   AsO�P   ?��AsP�   AsO�P   AsRPP   ?�xAsRu�   AsRPP   AsT�P   ?t�AsT��   AsT�P   AsW2P   ?�S(AsWW�   AsW2P   AsY�P   ?��@AsY��   AsY�P   As\P   ?���As\9�   As\P   As^�P   ?R�As^��   As^�P   As`�P   ?�w�Asa�   As`�P   AscgP   ?u"Asc��   AscgP   Ase�P   ?��*Ase��   Ase�P   AshIP   ?���Ashn�   AshIP   Asj�P   ?�^Asj��   Asj�P   Asm+P   ?�%�AsmP�   Asm+P   Aso�P   ?���Aso��   Aso�P   AsrP   ?�a�Asr2�   AsrP   Ast~P   ?���Ast��   Ast~P   Asv�P   ?���Asw�   Asv�P   Asy`P   ?L�Asy��   Asy`P   As{�P   ?��As{��   As{�P   As~BP   ?��As~g�   As~BP   As��P   ?�LAs���   As��P   As�$P   ?�9�As�I�   As�$P   As��P   ?�PAs���   As��P   As�P   ?�K�As�+�   As�P   As�wP   ?�X2As���   As�wP   As��P   ?��_As��   As��P   As�YP   ?�IAs�~�   As�YP   As��P   ?���As���   As��P   As�;P   ?��SAs�`�   As�;P   As��P   ?�8�As���   As��P   As�P   @�vAs�B�   As�P   As��P   ?���As���   As��P   As��P   ?�FMAs�$�   As��P   As�pP   ?�+FAs���   As�pP   As��P   ?�.�As��   As��P   As�RP   ?�(^As�w�   As�RP   As��P   ?�2�As���   As��P   As�4P   ?�X�As�Y�   As�4P   As��P   @��As���   As��P   As�P   ?��
As�;�   As�P   As��P   ?ªAs���   As��P   As��P   ?�&�As��   As��P   As�iP   ?�n'As���   As�iP   As��P   @oAs���   As��P   As�KP   @��As�p�   As�KP   As��P   ?�rVAs���   As��P   As�-P   @��As�R�   As�-P   AsP   ?���As���   AsP   As�P   @
kAs�4�   As�P   AsǀP   @�Asǥ�   AsǀP   As��P   ?���As��   As��P   As�bP   @�+Aṡ�   As�bP   As��P   @�<As���   As��P   As�DP   ?���As�i�   As�DP   AsӵP   ?�'�As���   AsӵP   As�&P   @9�As�K�   As�&P   AsؗP   @��Asؼ�   AsؗP   As�P   @y�As�-�   As�P   As�yP   @_�Asݞ�   As�yP   As��P   @%As��   As��P   As�[P   ?��As��   As�[P   As��P   @)PAs���   As��P   As�=P   @�AAs�b�   As�=P   As�P   @�	As���   As�P   As�P   @��As�D�   As�P   As�P   @ĜAs��   As�P   As�P   @:�kAs�&�   As�P   As�rP   @3UAs��   As�rP   As��P   @�?As��   As��P   As�TP   @��As�y�   As�TP   As��P   @��As���   As��P   As�6P   @"��As�[�   As�6P   As��P   @ЖAs���   As��P   AtP   @:%;At=�   AtP   At�P   @��At��   At�P   At�P   @"�#At�   At�P   At	kP   @��