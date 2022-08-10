CDF  �   
      time       bnds      lon       lat          .   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       EMRI-ESM2.0 (2017): 
aerosol: MASINGAR mk2r4 (TL95; 192 x 96 longitude/latitude; 80 levels; top level 0.01 hPa)
atmos: MRI-AGCM3.5 (TL159; 320 x 160 longitude/latitude; 80 levels; top level 0.01 hPa)
atmosChem: MRI-CCM2.1 (T42; 128 x 64 longitude/latitude; 80 levels; top level 0.01 hPa)
land: HAL 1.0
landIce: none
ocean: MRI.COM4.4 (tripolar primarily 0.5 deg latitude/1 deg longitude with meridional refinement down to 0.3 deg within 10 degrees north and south of the equator; 360 x 364 longitude/latitude; 61 levels; top grid cell 0-2 m)
ocnBgchem: MRI.COM4.4
seaIce: MRI.COM4.4      institution       CMeteorological Research Institute, Tsukuba, Ibaraki 305-0052, Japan    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2019-11-14T19:20:19Z   data_specs_version        01.00.31   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Lhttps://furtherinfo.es-doc.org/CMIP6.MRI.MRI-ESM2-0.historical.none.r1i1p1f1   grid      4native ocean tri-polar grid with 360x363 ocean cells   
grid_label        gn     history      
�Wed Aug 10 15:21:35 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/hfds/MRI-ESM2-0_r1i1p1f1/hfds_MRI-ESM2-0_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/MRI-ESM2-0_r1i1p1f1/CMIP6.ScenarioMIP.MRI.MRI-ESM2-0.ssp126.r1i1p1f1.Omon.hfds.gn.v20210329/hfds_Omon_MRI-ESM2-0_ssp126_r1i1p1f1_gn_201501-210012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/MRI-ESM2-0_r1i1p1f1/CMIP6.ScenarioMIP.MRI.MRI-ESM2-0.ssp126.r1i1p1f1.Omon.hfds.gn.v20210329/hfds_Omon_MRI-ESM2-0_ssp126_r1i1p1f1_gn_210101-230012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/hfds/MRI-ESM2-0_r1i1p1f1/hfds_MRI-ESM2-0_r1i1p1f1_ssp126.mergetime.nc
Wed Aug 10 15:21:35 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Omon.hfds.gn.v20210311/hfds_Omon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/hfds/MRI-ESM2-0_r1i1p1f1/hfds_MRI-ESM2-0_r1i1p1f1_historical.mergetime.nc
Fri Apr 08 04:23:18 2022: cdo -O -s -selname,hfds -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/hfds/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Omon.hfds.gn.v20210311/hfds_Omon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/hfds/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Omon.hfds.gn.v20210311/hfds_Omon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 04:23:10 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,hfds -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/hfds/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Omon.hfds.gn.v20210311/hfds_Omon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/hfds/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Omon.hfds.gn.v20210311/hfds_Omon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/hfds/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Omon.hfds.gn.v20210311/hfds_Omon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc
2019-11-14T19:20:19Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.      initialization_index            institution_id        MRI    mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      
MRI-ESM2-0     parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         ocean      	source_id         
MRI-ESM2-0     source_type       AOGCM AER CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        ACreation Date:(24 July 2019) MD5:c93735846d66458966fc81f390b2d714      title         $MRI-ESM2-0 output prepared for CMIP6   variable_id       hfds   variant_label         r1i1p1f1   license      CMIP6 model data produced by MRI is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.5.0      tracking_id       1hdl:21.14100/564cce55-e3c1-4241-8c17-3489c6612be1      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               T   	time_bnds                                 \   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               D   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               L   hfds                   	   standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any 'flux adjustment') .    cell_measures         area: areacello    history       �2019-11-14T19:20:19Z altered by CMOR: replaced missing value flag (-9.99e+33) and corresponding data with standard missing value (1e+20).               l                Aq���   Aq��P   Aq�P   >�t�Aq�6�   Aq�P   Aq��P   >ͯ�Aq���   Aq��P   Aq��P   >@PAq��   Aq��P   Aq�dP   >h��Aq���   Aq�dP   Aq��P   >��Aq���   Aq��P   Aq�FP   >��tAq�k�   Aq�FP   Aq��P   >_�Aq���   Aq��P   Aq�(P   ?��Aq�M�   Aq�(P   Aq��P   ?�Aq���   Aq��P   Aq�
P   >�c�Aq�/�   Aq�
P   Aq�{P   >�W�Aq���   Aq�{P   Aq��P   >]|nAq��   Aq��P   Aq�]P   ?�:AqĂ�   Aq�]P   Aq��P   >��Aq���   Aq��P   Aq�?P   >w")Aq�d�   Aq�?P   Aq˰P   >�\tAq���   Aq˰P   Aq�!P   ?��Aq�F�   Aq�!P   AqВP   ?v�Aqз�   AqВP   Aq�P   =���Aq�(�   Aq�P   Aq�tP   >�rNAqՙ�   Aq�tP   Aq��P   ?Aq�
�   Aq��P   Aq�VP   =x��Aq�{�   Aq�VP   Aq��P   >}�)Aq���   Aq��P   Aq�8P   ?	`Aq�]�   Aq�8P   Aq�P   ?xU�Aq���   Aq�P   Aq�P   >]7�Aq�?�   Aq�P   Aq�P   >���Aq��   Aq�P   Aq��P   ?J�hAq�!�   Aq��P   Aq�mP   =6rAq��   Aq�mP   Aq��P   >�0kAq��   Aq��P   Aq�OP   >���Aq�t�   Aq�OP   Aq��P   >��Aq���   Aq��P   Aq�1P   >5~�Aq�V�   Aq�1P   Aq��P   ��7Aq���   Aq��P   Aq�P   �<`pAq�8�   Aq�P   Aq��P   ?{Aq���   Aq��P   Aq��P   >�%Aq��   Aq��P   ArfP   >�XAr��   ArfP   Ar�P   >��Ar��   Ar�P   ArHP   ?[AArm�   ArHP   Ar�P   ?&�	Ar��   Ar�P   Ar*P   >'o�ArO�   Ar*P   Ar�P   ?��Ar��   Ar�P   ArP   ?AAr1�   ArP   Ar}P   >�Z~Ar��   Ar}P   Ar�P   ?�;Ar�   Ar�P   Ar_P   >�Z�Ar��   Ar_P   Ar�P   >���Ar��   Ar�P   ArAP   >�TZArf�   ArAP   Ar�P   ��QAr��   Ar�P   Ar!#P   ?�"�Ar!H�   Ar!#P   Ar#�P   ?�xAr#��   Ar#�P   Ar&P   ���
Ar&*�   Ar&P   Ar(vP   �%-PAr(��   Ar(vP   Ar*�P   ?AAr+�   Ar*�P   Ar-XP   >PbAr-}�   Ar-XP   Ar/�P   >�wAr/��   Ar/�P   Ar2:P   >�v�Ar2_�   Ar2:P   Ar4�P   >���Ar4��   Ar4�P   Ar7P   >��Ar7A�   Ar7P   Ar9�P   �c��Ar9��   Ar9�P   Ar;�P   ?,O�Ar<#�   Ar;�P   Ar>oP   ��rAr>��   Ar>oP   Ar@�P   >���ArA�   Ar@�P   ArCQP   =#ZiArCv�   ArCQP   ArE�P   >��ArE��   ArE�P   ArH3P   ?ArHX�   ArH3P   ArJ�P   ?�ArJ��   ArJ�P   ArMP   ?&:ArM:�   ArMP   ArO�P   ?''RArO��   ArO�P   ArQ�P   >bzArR�   ArQ�P   ArThP   >�/ ArT��   ArThP   ArV�P   >�ΊArV��   ArV�P   ArYJP   �\�sArYo�   ArYJP   Ar[�P   ?/�Ar[��   Ar[�P   Ar^,P   >�Ar^Q�   Ar^,P   Ar`�P   ?Q�Ar`��   Ar`�P   ArcP   ?��Arc3�   ArcP   AreP   >�uHAre��   AreP   Arg�P   >��+Arh�   Arg�P   ArjaP   ?�.Arj��   ArjaP   Arl�P   ?({Arl��   Arl�P   AroCP   ?g~Aroh�   AroCP   Arq�P   >�E~Arq��   Arq�P   Art%P   ?o��ArtJ�   Art%P   Arv�P   >�mArv��   Arv�P   AryP   �Iu Ary,�   AryP   Ar{xP   ? IAr{��   Ar{xP   Ar}�P   =�AAr~�   Ar}�P   Ar�ZP   >���Ar��   Ar�ZP   Ar��P   ?.�qAr���   Ar��P   Ar�<P   >�-�Ar�a�   Ar�<P   Ar��P   ?�Ar���   Ar��P   Ar�P   ?��Ar�C�   Ar�P   Ar��P   >���Ar���   Ar��P   Ar� P   >>� Ar�%�   Ar� P   Ar�qP   ?(<�Ar���   Ar�qP   Ar��P   ?$t�Ar��   Ar��P   Ar�SP   >/�Ar�x�   Ar�SP   Ar��P   ?l��Ar���   Ar��P   Ar�5P   >XAr�Z�   Ar�5P   Ar��P   > u�Ar���   Ar��P   Ar�P   �B�Ar�<�   Ar�P   Ar��P   ?h6 Ar���   Ar��P   Ar��P   ?�7Ar��   Ar��P   Ar�jP   ��RAr���   Ar�jP   Ar��P   >�n�Ar� �   Ar��P   Ar�LP   ?N޽Ar�q�   Ar�LP   Ar��P   >AyAr���   Ar��P   Ar�.P   =BXAr�S�   Ar�.P   Ar��P   >?��Ar���   Ar��P   Ar�P   ?&ɣAr�5�   Ar�P   Ar��P   =kqAr���   Ar��P   Ar��P   �Q�1Ar��   Ar��P   Ar�cP   �#��Ar���   Ar�cP   Ar��P   >�ƶAr���   Ar��P   Ar�EP   >^+�Ar�j�   Ar�EP   ArĶP   >G�Ar���   ArĶP   Ar�'P   >pT8Ar�L�   Ar�'P   ArɘP   >���Arɽ�   ArɘP   Ar�	P   ?!i�Ar�.�   Ar�	P   Ar�zP   >]b�ArΟ�   Ar�zP   Ar��P   >�nhAr��   Ar��P   Ar�\P   �ȹArӁ�   Ar�\P   Ar��P   ?}�Ar���   Ar��P   Ar�>P   ���Ar�c�   Ar�>P   ArگP   >�;Ar���   ArگP   Ar� P   ?@�Ar�E�   Ar� P   ArߑP   ?n�PAr߶�   ArߑP   Ar�P   =�Ar�'�   Ar�P   Ar�sP   ?�2Ar��   Ar�sP   Ar��P   >��qAr�	�   Ar��P   Ar�UP   >�KLAr�z�   Ar�UP   Ar��P   �A!�Ar���   Ar��P   Ar�7P   ?B��Ar�\�   Ar�7P   Ar�P   ?F�Ar���   Ar�P   Ar�P   >J��Ar�>�   Ar�P   Ar��P   ?�Ar���   Ar��P   Ar��P   >��Ar� �   Ar��P   Ar�lP   ?=�2Ar���   Ar�lP   Ar��P   ?jxJAr��   Ar��P   Ar�NP   �}8`Ar�s�   Ar�NP   As�P   ���;As��   As�P   As0P   ?,��AsU�   As0P   As�P   ?�ۃAs��   As�P   As	P   >ϼvAs	7�   As	P   As�P   ?�^pAs��   As�P   As�P   ?� ^As�   As�P   AseP   ?`?�As��   AseP   As�P   ?As��   As�P   AsGP   ?�CdAsl�   AsGP   As�P   ?:�As��   As�P   As)P   ?�+�AsN�   As)P   As�P   ?���As��   As�P   AsP   ?�JAs0�   AsP   As!|P   ?��As!��   As!|P   As#�P   ?U�3As$�   As#�P   As&^P   ?�b�As&��   As&^P   As(�P   >��As(��   As(�P   As+@P   ?<ڎAs+e�   As+@P   As-�P   ?��]As-��   As-�P   As0"P   ?�r�As0G�   As0"P   As2�P   ?}?qAs2��   As2�P   As5P   ?���As5)�   As5P   As7uP   ?���As7��   As7uP   As9�P   ?��*As:�   As9�P   As<WP   ?b�kAs<|�   As<WP   As>�P   ?�
�As>��   As>�P   AsA9P   ?��kAsA^�   AsA9P   AsC�P   ?mAsC��   AsC�P   AsFP   ?�+TAsF@�   AsFP   AsH�P   ?�l]AsH��   AsH�P   AsJ�P   ?��~AsK"�   AsJ�P   AsMnP   ?���AsM��   AsMnP   AsO�P   ?�c|AsP�   AsO�P   AsRPP   ?��AsRu�   AsRPP   AsT�P   ?���AsT��   AsT�P   AsW2P   ?�?�AsWW�   AsW2P   AsY�P   ?�1�AsY��   AsY�P   As\P   ?���As\9�   As\P   As^�P   ?��=As^��   As^�P   As`�P   ?�׽Asa�   As`�P   AscgP   ?�)Asc��   AscgP   Ase�P   ?�V�Ase��   Ase�P   AshIP   ?���Ashn�   AshIP   Asj�P   ?�Y�Asj��   Asj�P   Asm+P   @��AsmP�   Asm+P   Aso�P   ?Ϡ)Aso��   Aso�P   AsrP   ?�>�Asr2�   AsrP   Ast~P   ?�}XAst��   Ast~P   Asv�P   ?�OAsw�   Asv�P   Asy`P   ?�(�Asy��   Asy`P   As{�P   ?�T�As{��   As{�P   As~BP   ?�B�As~g�   As~BP   As��P   ?�|$As���   As��P   As�$P   ?��CAs�I�   As�$P   As��P   ?�B�As���   As��P   As�P   ?�As�+�   As�P   As�wP   ?���As���   As�wP   As��P   @$�As��   As��P   As�YP   @��As�~�   As�YP   As��P   ?�PYAs���   As��P   As�;P   @��As�`�   As�;P   As��P   ?���As���   As��P   As�P   ?�&+As�B�   As�P   As��P   ?�eQAs���   As��P   As��P   ?�beAs�$�   As��P   As�pP   ?��As���   As�pP   As��P   ?���As��   As��P   As�RP   ?�'�As�w�   As�RP   As��P   ?�As���   As��P   As�4P   ?��/As�Y�   As�4P   As��P   ?��As���   As��P   As�P   ?�o|As�;�   As�P   As��P   ?��As���   As��P   As��P   ?��GAs��   As��P   As�iP   ?�|As���   As�iP   As��P   ?ܧAs���   As��P   As�KP   ?��6As�p�   As�KP   As��P   ?�/@As���   As��P   As�-P   ?�EaAs�R�   As�-P   AsP   ?��EAs���   AsP   As�P   ?ݡ%As�4�   As�P   AsǀP   ?м<Asǥ�   AsǀP   As��P   ?��[As��   As��P   As�bP   ?�i�Aṡ�   As�bP   As��P   ?ڬ%As���   As��P   As�DP   ?�As�i�   As�DP   AsӵP   ?��9As���   AsӵP   As�&P   ?��]As�K�   As�&P   AsؗP   ?ŷ�Asؼ�   AsؗP   As�P   ?Ю�As�-�   As�P   As�yP   ?��Asݞ�   As�yP   As��P   ?�:�As��   As��P   As�[P   ?�O+As��   As�[P   As��P   ?��As���   As��P   As�=P   ?�"As�b�   As�=P   As�P   ?�keAs���   As�P   As�P   ?ͩ�As�D�   As�P   As�P   ?�Y�As��   As�P   As�P   ?�-�As�&�   As�P   As�rP   ?���As��   As�rP   As��P   ?�p�As��   As��P   As�TP   ?ܛsAs�y�   As�TP   As��P   ?���As���   As��P   As�6P   ?��As�[�   As�6P   As��P   ?���As���   As��P   AtP   ?��)At=�   AtP   At�P   ?�C�At��   At�P   At�P   ?�?nAt�   At�P   At	kP   ?f�<At	��   At	kP   At�P   ?�¨At�   At�P   AtMP   ?�1)Atr�   AtMP   At�P   ?�3fAt��   At�P   At/P   ?�w�AtT�   At/P   At�P   ?�BbAt��   At�P   AtP   ?w,�At6�   AtP   At�P   ?�j�At��   At�P   At�P   ?�*[At�   At�P   AtdP   ?���At��   AtdP   At!�P   ?ϊ�At!��   At!�P   At$FP   ?Ʊ�At$k�   At$FP   At&�P   ?�r�At&��   At&�P   At)(P   ?�?6At)M�   At)(P   At+�P   ?�(KAt+��   At+�P   At.
P   ?G�At./�   At.
P   At0{P   ?�&At0��   At0{P   At2�P   ?�	]At3�   At2�P   At5]P   ?���At5��   At5]P   At7�P   ?̴&At7��   At7�P   At:?P   ?���At:d�   At:?P   At<�P   ?��At<��   At<�P   At?!P   ?��At?F�   At?!P   AtA�P   ?���AtA��   AtA�P   AtDP   ?�SKAtD(�   AtDP   AtFtP   ?�2�AtF��   AtFtP   AtH�P   ?=�nAtI
�   AtH�P   AtKVP   ?���AtK{�   AtKVP   AtM�P   ?IJ�AtM��   AtM�P   AtP8P   ?�7�AtP]�   AtP8P   AtR�P   ?�D*AtR��   AtR�P   AtUP   ?��IAtU?�   AtUP   AtW�P   ?�CAtW��   AtW�P   AtY�P   ?�K�AtZ!�   AtY�P   At\mP   ?�]At\��   At\mP   At^�P   ?�}�At_�   At^�P   AtaOP   ?�'FAtat�   AtaOP   Atc�P   ?�NAtc��   Atc�P   Atf1P   ?��bAtfV�   Atf1P   Ath�P   ?�	cAth��   Ath�P   AtkP   ?���Atk8�   AtkP   Atm�P   ?�JAtm��   Atm�P   Ato�P   ?iIAtp�   Ato�P   AtrfP   ?��Atr��   AtrfP   Att�P   ?_��Att��   Att�P   AtwHP   ?���Atwm�   AtwHP   Aty�P   ?� �Aty��   Aty�P   At|*P   ?���At|O�   At|*P   At~�P   ?z��At~��   At~�P   At�P   ?EWAt�1�   At�P   At�}P   ?��{At���   At�}P   At��P   ?�e`At��   At��P   At�_P   ?];At���   At�_P   At��P   ?�۰At���   At��P   At�AP   ?��At�f�   At�AP   At��P   ?�־At���   At��P   At�#P   ?�jAt�H�   At�#P   At��P   ?��At���   At��P   At�P   ?���At�*�   At�P   At�vP   ?�TAt���   At�vP   At��P   ?�:�At��   At��P   At�XP   >���At�}�   At�XP   At��P   ?GT At���   At��P   At�:P   ?��vAt�_�   At�:P   At��P   ?�FZAt���   At��P   At�P   ?�>�At�A�   At�P   At��P   ?�l�At���   At��P   At��P   ?��At�#�   At��P   At�oP   ?w7kAt���   At�oP   At��P   ?�tAt��   At��P   At�QP   ?g?�At�v�   At�QP   At��P   ?�(�At���   At��P   At�3P   ?�.�At�X�   At�3P   At��P   ?��At���   At��P   At�P   ?�NAt�:�   At�P   At��P   ?�(�At���   At��P   At��P   ?���At��   At��P   At�hP   ?WAtō�   At�hP   At��P   ?��At���   At��P   At�JP   ?���At�o�   At�JP   At̻P   ?���At���   At̻P   At�,P   ?Z�At�Q�   At�,P   AtѝP   ?�1�At���   AtѝP   At�P   ?x�At�3�   At�P   At�P   ?�P�At֤�   At�P   At��P   ?�dAt��   At��P   At�aP   ?��qAtۆ�   At�aP   At��P   ?	��At���   At��P   At�CP   ?�b�At�h�   At�CP   At�P   ?��	At���   At�P   At�%P   ?�`wAt�J�   At�%P   At�P   ?M�oAt��   At�P   At�P   ?���At�,�   At�P   At�xP   ?{]AAt��   At�xP   At��P   ?���At��   At��P   At�ZP   ?�OuAt��   At�ZP   At��P   ?���At���   At��P   At�<P   ?��XAt�a�   At�<P   At��P   ?u�At���   At��P   At�P   ?�ĖAt�C�   At�P   At��P   ?�At���   At��P   Au  P   ?��Au %�   Au  P   AuqP   ?IAu��   AuqP   Au�P   ?��3Au�   Au�P   AuSP   ?�^9Aux�   AuSP   Au	�P   ?f�;Au	��   Au	�P   Au5P   ?�B�AuZ�   Au5P   Au�P   ?\�[Au��   Au�P   AuP   ?�ܾAu<�   AuP   Au�P   ?{:qAu��   Au�P   Au�P   ?�iAu�   Au�P   AujP   ?�{(Au��   AujP   Au�P   ?apXAu �   Au�P   AuLP   ?�BAuq�   AuLP   Au�P   ?UV�Au��   Au�P   Au".P   ? Au"S�   Au".P   Au$�P   ?�jAu$��   Au$�P   Au'P   ?��Au'5�   Au'P   Au)�P   ?�8�Au)��   Au)�P   Au+�P   ?�ߚAu,�   Au+�P   Au.cP   ?�x�Au.��   Au.cP   Au0�P   >� �Au0��   Au0�P   Au3EP   ?���Au3j�   Au3EP   Au5�P   ?�J-Au5��   Au5�P   Au8'P   ?�NAu8L�   Au8'P   Au:�P   ?^�&Au:��   Au:�P   Au=	P   ?Jx�Au=.�   Au=	P   Au?zP   ?]R�Au?��   Au?zP   AuA�P   ?{�7AuB�   AuA�P   AuD\P   ?� AuD��   AuD\P   AuF�P   ?L��AuF��   AuF�P   AuI>P   ?��AuIc�   AuI>P   AuK�P   ?��AuK��   AuK�P   AuN P   ?��AuNE�   AuN P   AuP�P   ?�?hAuP��   AuP�P   AuSP   ?�`�AuS'�   AuSP   AuUsP   ?��AuU��   AuUsP   AuW�P   >��AuX	�   AuW�P   AuZUP   ?C̗AuZz�   AuZUP   Au\�P   ?�Au\��   Au\�P   Au_7P   ?���Au_\�   Au_7P   Aua�P   ?�i�Aua��   Aua�P   AudP   ?zR�Aud>�   AudP   Auf�P   ?�8�Auf��   Auf�P   Auh�P   ?�ۅAui �   Auh�P   AuklP   ?��Auk��   AuklP   Aum�P   ?g��Aun�   Aum�P   AupNP   ?�2CAups�   AupNP   Aur�P   ?�D�Aur��   Aur�P   Auu0P   ?��|AuuU�   Auu0P   Auw�P   ?4Auw��   Auw�P   AuzP   ?#�WAuz7�   AuzP   Au|�P   ?���Au|��   Au|�P   Au~�P   ?A��Au�   Au~�P   Au�eP   ?/2Au���   Au�eP   Au��P   ?�o�Au���   Au��P   Au�GP   ?��YAu�l�   Au�GP   Au��P   ?z=+Au���   Au��P   Au�)P   ?Ю�Au�N�   Au�)P   Au��P   ?xPAu���   Au��P   Au�P   ?>$�Au�0�   Au�P   Au�|P   ?[Z'Au���   Au�|P   Au��P   ?l�jAu��   Au��P   Au�^P   ?}[PAu���   Au�^P   Au��P   ?�_�Au���   Au��P   Au�@P   ?�tzAu�e�   Au�@P   Au��P   ?� 9Au���   Au��P   Au�"P   ?��$Au�G�   Au�"P   Au��P   ?���Au���   Au��P   Au�P   >��9Au�)�   Au�P   Au�uP   ?�)�Au���   Au�uP   Au��P   ?3�GAu��   Au��P   Au�WP   ?TCDAu�|�   Au�WP   Au��P   >�1Au���   Au��P   Au�9P   ?��AAu�^�   Au�9P   Au��P   ?��Au���   Au��P   Au�P   ?@�PAu�@�   Au�P   Au��P   ?���Au���   Au��P   Au��P   ?�6�Au�"�   Au��P   Au�nP   ?d�Au���   Au�nP   Au��P   >�%�Au��   Au��P   Au�PP   ?��Au�u�   Au�PP   Au��P   ?��nAu���   Au��P   Au�2P   >Au�W�   Au�2P   AuʣP   ?�R�Au���   AuʣP   Au�P   ?t��Au�9�   Au�P   AuυP   ?�S[AuϪ�   AuυP   Au��P   ?p.*Au��   Au��P   Au�gP   ?I �AuԌ�   Au�gP   Au��P   ?,�MAu���   Au��P   Au�IP   ?��ZAu�n�   Au�IP   AuۺP   ?�T�Au���   AuۺP   Au�+P   ?�ޓAu�P�   Au�+P   Au��P   ?<ҫAu���   Au��P   Au�P   ?W��Au�2�   Au�P   Au�~P   ?5�Au��   Au�~P   Au��P   ?.^�Au��   Au��P   Au�`P   ?cBAu��   Au�`P   Au��P   ?�Au���   Au��P   Au�BP   ?�y�Au�g�   Au�BP   Au�P   ?N��